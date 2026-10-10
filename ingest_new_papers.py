"""Auditable batch ingestion for new CO2RR papers.

The command is dry-run by default.  It extracts one evidence-grounded JSON
record per PDF with local LoRA first (or a configured compatible API), optionally
asks a second model to review it, validates every product/FE/potential tuple, and
writes an audit bundle.  ``--apply`` appends only accepted, non-conflicting
records to the structured database and knowledge base.

Examples:
    python ingest_new_papers.py --pdf-dir incoming_papers
    python ingest_new_papers.py --pdf-dir incoming_papers --apply
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import pymupdf as fitz
except ImportError:  # Compatibility with older PyMuPDF packages.
    import fitz
import pandas as pd

from repair_paper_metadata import choose_doi, normalize_doi


PRODUCT_ALIASES = {
    "formate": "HCOOH", "formic acid": "HCOOH", "hcoo-": "HCOOH",
    "hcoo−": "HCOOH", "hcooh": "HCOOH", "carbon monoxide": "CO",
    "co": "CO", "methane": "CH4", "ch4": "CH4", "ethylene": "C2H4",
    "c2h4": "C2H4", "ethanol": "C2H5OH", "c2h5oh": "C2H5OH",
    "methanol": "CH3OH", "ch3oh": "CH3OH", "ethane": "C2H6",
    "c2h6": "C2H6", "acetate": "CH3COOH", "acetic acid": "CH3COOH",
    "propanol": "C3H7OH", "n-propanol": "C3H7OH", "syngas": "syngas",
}
REQUIRED_PRODUCT_FIELDS = {
    "product", "fe_pct", "voltage_V", "voltage_reference", "rank",
    "is_representative", "evidence_text", "evidence_section", "confidence",
}
RECORD_COLUMNS = [
    "doi", "title", "year", "catalyst_name", "catalyst_type", "product_raw",
    "product", "fe_pct", "voltage_V", "electrolyte", "cell_setup",
    "current_density_mA_cm2", "synthesis_method", "product_rank",
    "is_representative", "voltage_reference", "evidence_text",
    "evidence_section", "extraction_model", "review_model", "review_status",
    "ingested_at", "source_pdf",
]


EXTRACTION_SYSTEM = """You are a meticulous electrochemistry data curator.
Extract only claims explicitly supported by the supplied paper text. Never
infer a number, sign, product, or potential. Bind each product, FE and potential
to one experimental data point. The representative product is the product the
authors explicitly emphasize as the main/high-selectivity outcome; otherwise
use the highest reported FE for the studied catalyst. Other products are
secondary. Return JSON only."""

EXTRACTION_PROMPT = """Extract this CO2-reduction paper into this exact schema:
{{
  "doi": "", "title": "", "year": null,
  "catalyst_name": "", "catalyst_type": "",
  "electrolyte": "", "cell_setup": "",
  "current_density_mA_cm2": null, "synthesis_method": "",
  "products": [{{
    "product": "", "fe_pct": null, "voltage_V": null,
    "voltage_reference": "RHE|SHE|NHE|Ag/AgCl|SCE|unknown",
    "rank": "primary|secondary", "is_representative": true,
    "evidence_text": "verbatim sentence or compact table/caption excerpt",
    "evidence_section": "abstract|results|conclusion|table|figure_caption",
    "confidence": "high|medium|low"
  }}]
}}

Rules:
- FE is percentage 0-100, not a fraction.
- Preserve the potential sign exactly. Use null when the matching potential is
  not stated; never borrow a potential from another product.
- Include at most one representative product. Include secondary products only
  when their own FE and evidence are explicit.
- Do not use values from cited prior work, control catalysts, or references.
- evidence_text must contain the product and FE; it should contain the matching
  potential whenever voltage_V is non-null.

PAPER TEXT:
---
{text}
---"""

REVIEW_SYSTEM = """You audit structured CO2RR data against source evidence.
Reject unsupported FE values, wrong potential signs, product mismatches,
control-catalyst values, and secondary products marked representative. Return
JSON only and do not introduce facts absent from the supplied text."""

REVIEW_PROMPT = """Review the candidate against the paper text. Return:
{{"decision":"accept|revise|reject","issues":["..."],"corrected":<full schema or null>}}
Use "revise" only when every correction is directly supported by the text.

CANDIDATE:
{candidate}

PAPER TEXT:
---
{text}
---"""


def extract_pdf_text(path: Path) -> Tuple[str, int]:
    """Return embedded PDF text and page count."""
    with fitz.open(path) as doc:
        pages = [page.get_text("text") for page in doc]
    return "\n\n".join(pages), len(pages)


def compact_paper_text(text: str, max_chars: int = 55000) -> str:
    """Keep front matter, FE contexts and conclusions within an API budget."""
    clean = re.sub(r"[ \t]+", " ", text.replace("\x00", ""))
    if len(clean) <= max_chars:
        return clean
    pieces = [clean[:18000]]
    patterns = [r"faradaic", r"\bFE\b", r"selectiv", r"conclusion"]
    spans: List[Tuple[int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
            spans.append((max(0, match.start() - 850), min(len(clean), match.end() + 1200)))
    for start, end in sorted(spans):
        excerpt = clean[start:end]
        if excerpt not in pieces:
            pieces.append(excerpt)
        if sum(len(piece) for piece in pieces) >= max_chars - 9000:
            break
    pieces.append(clean[-8000:])
    return "\n\n[...section boundary...]\n\n".join(pieces)[:max_chars]


def load_config(path: Path) -> Dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if "primary" not in config:
        raise ValueError("Model config must define a primary profile")
    for name in ('primary', 'fallback', 'reviewer'):
        profile = config.get(name)
        if profile and profile.get('backend', 'api') not in ('api', 'local_lora'):
            raise ValueError(f'Unsupported {name} backend')
        if name != 'primary' and profile and profile.get('backend', 'api') != 'api':
            raise ValueError(f'{name} currently requires an API profile')
    return config


def _secret(profile: Dict[str, Any]) -> str:
    env_name = profile.get("api_key_env", "")
    value = os.environ.get(env_name, "") if env_name else ""
    if not value:
        raise ValueError(f"Missing API key (environment variable: {env_name or 'not configured'})")
    return value


def call_chat_api(
    profile: Dict[str, Any], system: str, user: str, retries: int = 3
) -> Tuple[str, Dict[str, Any]]:
    """Call an OpenAI-compatible /chat/completions endpoint using stdlib."""
    base_url = os.getenv(profile.get('base_url_env', ''), '') or str(profile["base_url"])
    base_url = base_url.rstrip("/")
    endpoint = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
    payload = {
        "model": profile["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": profile.get("temperature", 0),
        "max_tokens": profile.get("max_tokens", 3500),
    }
    if profile.get("response_format", True):
        payload["response_format"] = {"type": "json_object"}
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {_secret(profile)}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            context = ssl.create_default_context()
            with urlopen(request, timeout=int(profile.get("timeout", 180)), context=context) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            return content, usage
        except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"API failed after {retries} attempts: {last_error}")


def parse_json_response(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def normalize_product(value: str) -> str:
    raw = str(value or "").strip()
    return PRODUCT_ALIASES.get(raw.lower(), raw)


def _number_in_evidence(value: float, evidence: str, tolerance: float = 0.011) -> bool:
    evidence = evidence.replace("−", "-").replace("–", "-").replace("—", "-")
    numbers = [float(item) for item in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", evidence)]
    return any(abs(number - value) <= tolerance for number in numbers)


def _evidence_in_source(evidence: str, source_text: str) -> bool:
    def normalize(value: str) -> str:
        value = value.replace("−", "-").replace("–", "-")
        return re.sub(r"\s+", " ", value).strip().lower()
    evidence_normalized = normalize(evidence)
    source_normalized = normalize(source_text)
    if evidence_normalized in source_normalized:
        return True
    tokens = [
        token for token in re.findall(r"[a-z0-9.%+-]+", evidence_normalized)
        if len(token) > 1
    ]
    if len(tokens) < 5:
        return False
    return sum(token in source_normalized for token in tokens) / len(tokens) >= 0.85


def validate_candidate(
    candidate: Dict[str, Any], source_doi: str, source_text: str = ""
) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize a model candidate and return blocking validation issues."""
    issues: List[str] = []
    result = dict(candidate)
    result["doi"] = normalize_doi(result.get("doi") or source_doi)
    if source_doi and result["doi"] != source_doi:
        issues.append(f"DOI mismatch: source={source_doi}, extracted={result['doi']}")
    for field in ("title", "catalyst_name"):
        if not str(result.get(field, "")).strip():
            issues.append(f"missing {field}")
    products = result.get("products")
    if not isinstance(products, list) or not products:
        issues.append("no product records")
        result["products"] = []
        return result, issues
    representatives = 0
    normalized_products = []
    for index, item in enumerate(products):
        if not isinstance(item, dict):
            issues.append(f"product[{index}] is not an object")
            continue
        missing = REQUIRED_PRODUCT_FIELDS.difference(item)
        if missing:
            issues.append(f"product[{index}] missing {sorted(missing)}")
        normalized = dict(item)
        normalized["product"] = normalize_product(item.get("product", ""))
        try:
            normalized["fe_pct"] = float(item.get("fe_pct"))
        except (TypeError, ValueError):
            issues.append(f"product[{index}] invalid FE")
            continue
        if not 0 < normalized["fe_pct"] <= 100:
            issues.append(f"product[{index}] FE outside (0,100]")
        voltage = item.get("voltage_V")
        try:
            normalized["voltage_V"] = None if voltage is None else float(voltage)
        except (TypeError, ValueError):
            issues.append(f"product[{index}] invalid voltage")
            normalized["voltage_V"] = None
        evidence = str(item.get("evidence_text", "")).strip()
        if len(evidence) < 18:
            issues.append(f"product[{index}] evidence too short")
        if not _number_in_evidence(normalized["fe_pct"], evidence):
            issues.append(f"product[{index}] FE absent from evidence")
        if normalized["voltage_V"] is not None and not _number_in_evidence(normalized["voltage_V"], evidence):
            issues.append(f"product[{index}] voltage absent from evidence")
        if source_text and not _evidence_in_source(evidence, source_text):
            issues.append(f"product[{index}] evidence not found in source text")
        representative = bool(item.get("is_representative"))
        if representative:
            representatives += 1
            normalized["rank"] = "primary"
        elif item.get("rank") == "primary":
            issues.append(f"product[{index}] primary rank without representative flag")
        normalized_products.append(normalized)
    if representatives != 1:
        issues.append(f"expected exactly one representative product, found {representatives}")
    result["products"] = normalized_products
    return result, issues


def core_signature(candidate: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    rows = []
    for product in candidate.get("products", []):
        rows.append((
            normalize_product(product.get("product", "")),
            round(float(product.get("fe_pct", -1)), 2),
            None if product.get("voltage_V") is None else round(float(product["voltage_V"]), 3),
            bool(product.get("is_representative")),
        ))
    return sorted(rows, key=str)


def _mentions(value: str, text: str) -> bool:
    return bool(re.search(r'(?<![A-Za-z0-9])' + re.escape(value)
                          + r'(?![A-Za-z0-9])', text, flags=re.IGNORECASE))


def lora_to_candidate(
    outputs: List[Dict], source_doi: str, pdf_name: str,
) -> Dict:
    """Bind legacy LoRA arrays using source evidence, never by array position."""
    materials = {
        material.strip() for output in outputs
        for material in output['result']['materials']
        if material.strip() and _mentions(material.strip(), output['source_text'])
    }
    if len(materials) != 1:
        raise ValueError('LoRA 未明确识别唯一催化剂，需 API 补充或人工审核。')
    catalyst = next(iter(materials))
    products = {}
    for output in outputs:
        prediction = output['result']
        normalized = {normalize_product(value) for value in prediction['products'] if value.strip()}
        # Same-sentence evidence is conservative: ambiguous cross-sentence data go to review.
        sentences = re.split(
            r'(?<!vs\.)(?<!Vs\.)(?<=[.!?])\s+(?=[A-Z])|\n\s*\n',
            output['source_text'],
        )
        for efficiency in prediction['efficiency']:
            fe_match = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)\s*%', efficiency)
            if not fe_match:
                continue
            fe = float(fe_match.group(1))
            if not 0 < fe <= 100:
                continue
            voltage_match = re.search(r'\bat\s+([+−–-]?\d+(?:\.\d+)?)\s*V\b', efficiency, re.I)
            voltage = float(voltage_match.group(1).replace('−', '-').replace('–', '-')) if voltage_match else None
            reference_match = re.search(r'\bvs\.?\s*(RHE|SHE|NHE|Ag/AgCl|SCE)\b', efficiency, re.I)
            reference = reference_match.group(1) if reference_match else 'unknown'
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 18 or not _mentions(catalyst, sentence):
                    continue
                percentages = re.findall(r'(?<![\d.])(\d+(?:\.\d+)?)\s*%', sentence)
                # Multiple different FE values or predicted products are not safely alignable.
                if not percentages or any(abs(float(value) - fe) > 0.011 for value in percentages):
                    continue
                matched = []
                for product in normalized:
                    aliases = {product} | {alias for alias, canonical in PRODUCT_ALIASES.items() if canonical == product}
                    if any(_mentions(alias, sentence) for alias in aliases):
                        matched.append(product)
                if len(matched) != 1:
                    continue
                source_products = {
                    canonical for alias, canonical in PRODUCT_ALIASES.items()
                    if _mentions(alias, sentence) or _mentions(canonical, sentence)
                }
                if source_products.difference(matched):
                    continue
                if voltage is not None:
                    measured = re.findall(r'([+−–-]?\d+(?:\.\d+)?)\s*V\b', sentence)
                    if not any(abs(float(value.replace('−', '-').replace('–', '-')) - voltage) <= 0.011 for value in measured):
                        continue
                    if reference != 'unknown' and not _mentions(reference, sentence):
                        continue
                product = matched[0]
                key = (product, fe, voltage)
                products[key] = {
                    'product': product, 'fe_pct': fe, 'voltage_V': voltage,
                    'voltage_reference': reference, 'rank': 'secondary',
                    'is_representative': False, 'evidence_text': sentence,
                    'evidence_section': 'paper_text', 'confidence': 'medium',
                }
    if not products:
        raise ValueError('LoRA 提取结果缺少可核对的催化剂、产物和 FE 原文绑定，需 API 补充或人工审核。')
    values = list(products.values())
    representative = max(values, key=lambda item: item['fe_pct'])
    representative.update(is_representative=True, rank='primary')
    return {
        'doi': source_doi, 'title': Path(pdf_name).stem.replace('_', ' '),
        'title_source': 'pdf_filename', 'catalyst_name': catalyst,
        'catalyst_type': 'Other', 'year': None, 'electrolyte': 'unknown',
        'cell_setup': '', 'current_density_mA_cm2': None,
        'synthesis_method': '', 'products': values,
    }


def run_extractor(
    profile: Dict, prompt: str, text: str, source_doi: str, pdf_name: str,
) -> Tuple[Dict, Dict]:
    if profile.get('backend', 'api') == 'local_lora':
        from local_lora_extractor import extract_with_lora
        outputs, usage = extract_with_lora(profile, text)
        return lora_to_candidate(outputs, source_doi, pdf_name), usage
    raw, usage = call_chat_api(profile, EXTRACTION_SYSTEM, prompt)
    return parse_json_response(raw), usage


def candidate_to_rows(
    candidate: Dict[str, Any], pdf_name: str, primary_model: str,
    reviewer_model: str, review_status: str,
) -> List[Dict[str, Any]]:
    common = {
        "doi": normalize_doi(candidate.get("doi", "")),
        "title": candidate.get("title", ""), "year": candidate.get("year"),
        "catalyst_name": candidate.get("catalyst_name", ""),
        "catalyst_type": candidate.get("catalyst_type", "Other"),
        "electrolyte": candidate.get("electrolyte", "unknown"),
        "cell_setup": candidate.get("cell_setup", ""),
        "current_density_mA_cm2": candidate.get("current_density_mA_cm2"),
        "synthesis_method": candidate.get("synthesis_method", ""),
        "extraction_model": primary_model, "review_model": reviewer_model,
        "review_status": review_status,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "source_pdf": pdf_name,
    }
    rows = []
    for product in candidate.get("products", []):
        row = dict(common)
        row.update({
            "product_raw": product["product"], "product": product["product"],
            "fe_pct": product["fe_pct"], "voltage_V": product.get("voltage_V"),
            "product_rank": product.get("rank", "secondary"),
            "is_representative": bool(product.get("is_representative")),
            "voltage_reference": product.get("voltage_reference", "unknown"),
            "evidence_text": product.get("evidence_text", ""),
            "evidence_section": product.get("evidence_section", ""),
        })
        rows.append(row)
    return rows


def existing_keys(records_path: Path) -> Tuple[set, set]:
    if not records_path.exists():
        return set(), set()
    frame = pd.read_csv(records_path)
    dois = {normalize_doi(value) for value in frame.get("doi", [])}
    keys = {
        (normalize_doi(row.get("doi", "")), str(row.get("catalyst_name", "")).lower(), str(row.get("product", "")).lower())
        for _, row in frame.iterrows()
    }
    return dois, keys


def write_audit(out_dir: Path, audits: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ingestion_audit.jsonl").open("w", encoding="utf-8") as handle:
        for audit in audits:
            handle.write(json.dumps(audit, ensure_ascii=False, default=str) + "\n")
    summary_fields = [
        "pdf_file", "doi", "status", "reason", "primary_model",
        "review_model", "text_source", "pages", "text_chars",
    ]
    with (out_dir / "ingestion_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({field: audit.get(field, "") for field in summary_fields} for audit in audits)
    pd.DataFrame(rows, columns=RECORD_COLUMNS).to_csv(out_dir / "accepted_records.csv", index=False)


def apply_records(records_path: Path, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    current = pd.read_csv(records_path) if records_path.exists() else pd.DataFrame()
    for column in RECORD_COLUMNS:
        if column not in current.columns:
            current[column] = ""
    additions = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    combined = additions if current.empty else pd.concat(
        [current, additions], ignore_index=True, sort=False
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if records_path.exists():
        shutil.copy2(records_path, records_path.with_name(f"{records_path.stem}.before-ingest-{stamp}.csv"))
    combined.to_csv(records_path, index=False)
    return len(additions)


def apply_knowledge_base(kb_path: Path, accepted: List[Dict[str, Any]]) -> int:
    entries = json.loads(kb_path.read_text(encoding="utf-8")) if kb_path.exists() else []
    existing = {normalize_doi(entry.get("doi", "")) for entry in entries}
    added = 0
    for item in accepted:
        candidate = item["candidate"]
        doi = normalize_doi(candidate.get("doi", ""))
        if doi in existing:
            continue
        entries.append({
            "pdf_file": item["pdf_file"], "full_text_chars": item["text_chars"],
            "sections": {
                "abstract": item.get("evidence_text", ""),
                "results": item.get("paper_excerpt", ""),
                "conclusion": "",
            },
            "summary": item.get("evidence_text", ""), "records": item["rows"],
            "doi": doi, "doi_source": "batch_ingestion", "doi_confidence": "high",
        })
        existing.add(doi)
        added += 1
    if added:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if kb_path.exists():
            shutil.copy2(kb_path, kb_path.with_name(f"{kb_path.stem}.before-ingest-{stamp}.json"))
        kb_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return added


def reconcile_duplicate_kb_dois(
    kb_path: Path, audits: List[Dict[str, Any]], structured_dois: set
) -> int:
    """Repair legacy KB DOI aliases when PDF-detected DOI already exists.

    Only audits classified as duplicate are eligible, and the replacement DOI
    must already be present in the curated structured database.
    """
    if not kb_path.exists():
        return 0
    replacements = {
        str(audit.get("pdf_file", "")): normalize_doi(audit.get("doi", ""))
        for audit in audits
        if audit.get("status") in {"accepted", "duplicate"}
        and normalize_doi(audit.get("doi", "")) in structured_dois
    }
    if not replacements:
        return 0
    entries = json.loads(kb_path.read_text(encoding="utf-8"))
    changed = 0
    for entry in entries:
        pdf_file = str(entry.get("pdf_file", ""))
        replacement = replacements.get(pdf_file)
        if not replacement or normalize_doi(entry.get("doi", "")) == replacement:
            continue
        entry["doi"] = replacement
        entry["doi_source"] = "pdf_detected_during_ingestion"
        entry["doi_confidence"] = "high"
        changed += 1
    # An earlier ingestion version appended a thin entry when the same PDF had
    # a legacy DOI. Keep the richest entry for each non-empty PDF filename.
    deduplicated: List[Dict[str, Any]] = []
    pdf_to_index: Dict[str, int] = {}
    for entry in entries:
        pdf_file = str(entry.get("pdf_file", ""))
        if not pdf_file:
            deduplicated.append(entry)
            continue
        if pdf_file not in pdf_to_index:
            pdf_to_index[pdf_file] = len(deduplicated)
            deduplicated.append(entry)
            continue
        previous_index = pdf_to_index[pdf_file]
        previous = deduplicated[previous_index]
        previous_size = int(previous.get("full_text_chars", 0) or 0)
        current_size = int(entry.get("full_text_chars", 0) or 0)
        if current_size > previous_size:
            deduplicated[previous_index] = entry
        changed += 1
    entries = deduplicated
    if changed:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(
            kb_path,
            kb_path.with_name(f"{kb_path.stem}.before-ingest-doi-reconcile-{stamp}.json"),
        )
        kb_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    return changed


def process_pdf(
    pdf_path: Path, config: Dict[str, Any], existing_dois: set,
    fixture: Optional[Dict[str, Any]] = None, ocr_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    text, pages = extract_pdf_text(pdf_path)
    text_source = "pdf_text"
    min_text_chars = int(config.get("min_text_chars", 1500))
    if ocr_dir is not None:
        ocr_path = ocr_dir / f"{pdf_path.stem}.md"
        if ocr_path.exists() and (
            config.get("prefer_ocr", False) or len(text.strip()) < min_text_chars
        ):
            text = ocr_path.read_text(encoding="utf-8", errors="replace")
            text_source = "ocr_markdown"
    source_doi, doi_confidence = choose_doi(text[:50000])
    audit: Dict[str, Any] = {
        "pdf_file": pdf_path.name, "doi": source_doi, "doi_confidence": doi_confidence,
        "pages": pages, "text_chars": len(text), "status": "review",
        "text_source": text_source,
        "reason": "", "primary_model": config["primary"].get("model", "fixture"),
        "review_model": (config.get("reviewer") or {}).get("model", ""),
    }
    if len(text.strip()) < min_text_chars:
        audit.update(status="review", reason="insufficient embedded text; OCR required")
        return audit
    if not source_doi or doi_confidence == "low":
        audit.update(status="review", reason="DOI missing or low confidence")
        return audit
    if source_doi in existing_dois:
        audit.update(status="duplicate", reason="DOI already exists")
        return audit

    compact = compact_paper_text(text, int(config.get("max_input_chars", 55000)))
    try:
        if fixture is not None:
            candidate = fixture
            audit["primary_usage"] = {}
            audit.update(primary_model='fixture', primary_backend='fixture', fallback_used=False)
        else:
            prompt = EXTRACTION_PROMPT.format(text=compact)
            if config.get("representative_only", False):
                prompt += (
                    "\n\nSTRICT REPRESENTATIVE-ONLY MODE: Return exactly one products item: "
                    "the author's explicitly emphasized main product at its best reported FE. "
                    "Do not include any secondary product. If no explicit main-product FE is "
                    "supported, return an empty products array."
                )
            primary = config['primary']
            backend = primary.get('backend', 'api')
            audit['primary_backend'] = backend
            audit['fallback_used'] = False
            try:
                candidate, usage = run_extractor(primary, prompt, compact, source_doi, pdf_path.name)
                candidate, primary_issues = validate_candidate(candidate, source_doi, compact)
                if backend == 'local_lora' and primary_issues:
                    raise ValueError('; '.join(primary_issues))
            except (RuntimeError, ValueError, OSError, ImportError, KeyError) as exc:
                if backend != 'local_lora':
                    raise
                audit['local_lora_error'] = str(exc)
                fallback = config.get('fallback')
                if not config.get('allow_api_fallback', False) or not fallback:
                    raise
                env_name = fallback.get('api_key_env', '')
                if not env_name or not os.getenv(env_name, '').strip():
                    audit['fallback_skipped'] = 'API key not configured'
                    raise
                audit['fallback_used'] = True
                audit['primary_backend'] = 'api'
                audit['primary_model'] = fallback['model']
                candidate, usage = run_extractor(fallback, prompt, compact, source_doi, pdf_path.name)
            audit['primary_usage'] = usage
            if audit['primary_backend'] == 'local_lora':
                audit['primary_model'] = usage.get('model', audit['primary_model'])
            if config.get('representative_only', False) and backend == 'local_lora' and not audit['fallback_used']:
                candidate['products'] = [item for item in candidate['products'] if item['is_representative']]
        candidate, issues = validate_candidate(candidate, source_doi, compact)
        audit["primary_candidate"] = candidate
        audit["primary_issues"] = issues
        if issues:
            audit.update(status="review", reason="; ".join(issues))
            return audit

        review_status = "single_model_validated"
        reviewer = config.get("reviewer")
        if reviewer and fixture is None:
            raw_review, usage = call_chat_api(
                reviewer, REVIEW_SYSTEM,
                REVIEW_PROMPT.format(candidate=json.dumps(candidate, ensure_ascii=False), text=compact),
            )
            review = parse_json_response(raw_review)
            audit["review"] = review
            audit["review_usage"] = usage
            decision = review.get("decision")
            if decision == "revise" and isinstance(review.get("corrected"), dict):
                corrected, correction_issues = validate_candidate(
                    review["corrected"], source_doi, compact
                )
                if correction_issues:
                    audit.update(status="review", reason="review correction invalid: " + "; ".join(correction_issues))
                    return audit
                candidate = corrected
                review_status = "reviewer_corrected"
            elif decision != "accept":
                audit.update(status="review", reason="reviewer decision: " + str(decision))
                return audit
            else:
                review_status = "reviewer_accepted"

        evidence = "\n".join(product.get("evidence_text", "") for product in candidate["products"])
        audit.update(status="accepted", reason="validated", candidate=candidate,
                     review_status=review_status, evidence_text=evidence,
                     paper_excerpt=compact)
        return audit
    except (RuntimeError, ValueError, OSError, ImportError, KeyError, json.JSONDecodeError) as exc:
        audit.update(status="error", reason=str(exc))
        return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-grounded batch ingestion of new CO2RR PDFs")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--config", default="config/ingest_models.lora.example.json")
    parser.add_argument("--output-dir", default="outputs/new_paper_ingestion")
    parser.add_argument("--ocr-dir", default="outputs/ocr_markdown")
    parser.add_argument("--records", default="outputs/structured_records.csv")
    parser.add_argument("--knowledge-base", default="outputs/knowledge_base.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--fixture-response", help="Offline JSON response for smoke testing one PDF")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse per-PDF results already present in output-dir/ingestion_audit.jsonl",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.limit is not None:
        pdfs = pdfs[:args.limit]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")
    config = load_config(Path(args.config))
    fixture = json.loads(Path(args.fixture_response).read_text(encoding="utf-8")) if args.fixture_response else None
    existing_dois, existing_record_keys = existing_keys(Path(args.records))
    output_dir = Path(args.output_dir)
    audits: List[Dict[str, Any]] = []
    audit_path = output_dir / "ingestion_audit.jsonl"
    if args.resume and audit_path.exists():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                audits.append(json.loads(line))
    completed = {audit.get("pdf_file") for audit in audits}
    for path in pdfs:
        if path.name in completed:
            continue
        audit = process_pdf(
            path, config, existing_dois,
            fixture if len(pdfs) == 1 else None, Path(args.ocr_dir),
        )
        audits.append(audit)
        write_audit(output_dir, audits, [])
        print(
            f"[{len(audits)}/{len(pdfs)}] {path.name}: "
            f"{audit['status']} - {audit['reason']}",
            flush=True,
        )

    rows: List[Dict[str, Any]] = []
    accepted_for_kb = []
    for audit in audits:
        if audit["status"] != "accepted":
            continue
        candidate = audit["candidate"]
        candidate_rows = candidate_to_rows(
            candidate, audit["pdf_file"], audit["primary_model"], audit["review_model"], audit["review_status"],
        )
        conflicts = [
            row for row in candidate_rows
            if (normalize_doi(row["doi"]), row["catalyst_name"].lower(), row["product"].lower()) in existing_record_keys
        ]
        if conflicts:
            audit.update(status="review", reason="record key conflicts with existing database")
            continue
        audit["rows"] = candidate_rows
        rows.extend(candidate_rows)
        accepted_for_kb.append(audit)

    write_audit(output_dir, audits, rows)
    accepted_count = sum(audit["status"] == "accepted" for audit in audits)
    print(f"PDFs: {len(pdfs)}; accepted: {accepted_count}; review/error/duplicate: {len(pdfs) - accepted_count}")
    print(f"Audit bundle: {args.output_dir}")
    if not args.apply:
        print("Dry run only. Review ingestion_summary.csv and accepted_records.csv, then re-run with --apply.")
        return
    added_rows = apply_records(Path(args.records), rows)
    added_kb = apply_knowledge_base(Path(args.knowledge_base), accepted_for_kb)
    current_dois, _ = existing_keys(Path(args.records))
    reconciled = reconcile_duplicate_kb_dois(
        Path(args.knowledge_base), audits, current_dois
    )
    print(
        f"Applied {added_rows} structured rows, {added_kb} knowledge-base entries, "
        f"and reconciled {reconciled} legacy KB DOI mappings."
    )


if __name__ == "__main__":
    main()
