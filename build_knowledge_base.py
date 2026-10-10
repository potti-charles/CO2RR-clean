"""Step 1: Build CO2RR catalyst knowledge base from PDFs + structured records.

Extracts key paragraphs from PDFs (abstract, synthesis, results, conclusion)
and combines with structured catalyst data for RAG retrieval.

Usage:
    python build_knowledge_base.py
    python build_knowledge_base.py --pdf_dir gold_corpus/pdfs --limit 10
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import fitz
import pandas as pd


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Extract full text from a PDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Section detection & key paragraph extraction
# ---------------------------------------------------------------------------

# Common section headers in chemistry papers
_SECTION_PATTERNS = {
    "abstract": re.compile(
        r"^(abstract|summary)\s*$", re.IGNORECASE | re.MULTILINE
    ),
    "introduction": re.compile(
        r"^[\d.]*\s*(introduction)\s*$", re.IGNORECASE | re.MULTILINE
    ),
    "experimental": re.compile(
        r"^[\d.]*\s*(experimental|methods|materials and methods|"
        r"synthesis|preparation|electrode preparation|"
        r"catalyst preparation|electrocatalyst preparation)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "results": re.compile(
        r"^[\d.]*\s*(results|results and discussion|"
        r"electrochemical|electrocatalytic|CO2 reduction|"
        r"catalytic performance|performance)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "conclusion": re.compile(
        r"^[\d.]*\s*(conclusions?|summary and conclusions?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
}

# Max chars per section to keep
_MAX_SECTION_CHARS = 2000


def extract_sections(text: str) -> Dict[str, str]:
    """Extract key sections from paper text.

    Returns dict with keys: abstract, experimental, results, conclusion.
    Each value is the extracted text (truncated to ~2000 chars).
    """
    sections = {}

    # Try to find abstract - often between "Abstract" and "Introduction"
    # or at the very beginning of the paper
    abs_match = _SECTION_PATTERNS["abstract"].search(text)
    intro_match = _SECTION_PATTERNS["introduction"].search(text)

    if abs_match:
        start = abs_match.end()
        end = intro_match.start() if intro_match and intro_match.start() > start else start + _MAX_SECTION_CHARS
        sections["abstract"] = text[start:end].strip()[:_MAX_SECTION_CHARS]
    else:
        # Heuristic: first 1500 chars often contain abstract-like content
        sections["abstract"] = text[:1500].strip()

    # Find other sections by scanning for headers
    for section_name in ("experimental", "results", "conclusion"):
        pattern = _SECTION_PATTERNS[section_name]
        match = pattern.search(text)
        if match:
            start = match.end()
            # Find the next section header after this one
            remaining = text[start:]
            next_header = re.search(
                r"^[\d.]*\s*(?:References|Acknowledgement|Supporting|Supplementary|"
                r"Author contributions|Declaration|Conflict|Notes)\s*$",
                remaining,
                re.IGNORECASE | re.MULTILINE,
            )
            if next_header:
                section_text = remaining[: next_header.start()]
            else:
                section_text = remaining[:_MAX_SECTION_CHARS]
            sections[section_name] = section_text.strip()[:_MAX_SECTION_CHARS]

    return sections


def extract_first_paragraphs(text: str, n_chars: int = 3000) -> str:
    """Fallback: just return the first n_chars of the paper."""
    return text[:n_chars].strip()


# ---------------------------------------------------------------------------
# Build knowledge base entries
# ---------------------------------------------------------------------------

def build_pdf_entry(pdf_path: Path) -> Optional[Dict]:
    """Build a knowledge base entry from a single PDF."""
    try:
        text = extract_text(str(pdf_path))
    except Exception as e:
        print(f"  ERROR reading {pdf_path.name}: {e}")
        return None

    if len(text.strip()) < 100:
        return None

    sections = extract_sections(text)

    # Build a combined summary for retrieval
    summary_parts = []
    if sections.get("abstract"):
        summary_parts.append(sections["abstract"])
    if sections.get("conclusion"):
        summary_parts.append(sections["conclusion"])

    entry = {
        "pdf_file": pdf_path.name,
        "full_text_chars": len(text),
        "sections": sections,
        "summary": "\n\n".join(summary_parts) if summary_parts else extract_first_paragraphs(text),
    }
    return entry


def link_records_to_pdfs(
    records_df: pd.DataFrame, pdf_entries: List[Dict]
) -> List[Dict]:
    """Link structured records to their PDF entries by DOI/title matching."""
    # Build DOI-to-records mapping
    doi_records = {}
    for _, row in records_df.iterrows():
        doi = row.get("doi", "")
        if doi:
            if doi not in doi_records:
                doi_records[doi] = []
            doi_records[doi].append({
                "catalyst_name": row["catalyst_name"],
                "catalyst_type": row.get("catalyst_type", ""),
                "product": row["product"],
                "fe_pct": row["fe_pct"],
                "voltage_V": row.get("voltage_V"),
                "electrolyte": row.get("electrolyte", ""),
            })

    # Try to match PDFs to DOIs via filename or content
    matched = 0
    for entry in pdf_entries:
        entry["records"] = []
        summary_lower = entry.get("summary", "").lower()

        for doi, recs in doi_records.items():
            # Check if DOI appears in the PDF text
            doi_short = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
            if doi_short.lower() in summary_lower or doi_short.lower() in entry.get("sections", {}).get("abstract", "").lower():
                entry["records"] = recs
                entry["doi"] = doi
                matched += 1
                break

        # Fallback: match by catalyst name in abstract
        if not entry["records"]:
            for doi, recs in doi_records.items():
                for rec in recs:
                    cat_name = rec["catalyst_name"].lower()
                    # Use first significant word of catalyst name
                    words = [w for w in cat_name.split() if len(w) > 3]
                    if words and any(w in summary_lower for w in words[:2]):
                        entry["records"] = recs
                        entry["doi"] = doi
                        matched += 1
                        break
                if entry["records"]:
                    break

    print(f"  Matched {matched}/{len(pdf_entries)} PDFs to structured records")
    return pdf_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build CO2RR knowledge base")
    parser.add_argument("--pdf_dir", default="gold_corpus/pdfs")
    parser.add_argument("--records_csv", default="outputs/structured_records.csv")
    parser.add_argument("--output", default="outputs/knowledge_base.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to process")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if args.limit:
        pdf_files = pdf_files[:args.limit]
    print(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    # Process PDFs
    pdf_entries = []
    for i, pdf_path in enumerate(pdf_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing [{i+1}/{len(pdf_files)}] {pdf_path.name[:60]}...")
        entry = build_pdf_entry(pdf_path)
        if entry:
            pdf_entries.append(entry)

    print(f"\nExtracted sections from {len(pdf_entries)}/{len(pdf_files)} PDFs")

    # Section coverage stats
    section_counts = {s: 0 for s in ("abstract", "experimental", "results", "conclusion")}
    for entry in pdf_entries:
        for s in section_counts:
            if entry["sections"].get(s):
                section_counts[s] += 1
    print(f"Section coverage:")
    for s, c in section_counts.items():
        print(f"  {s}: {c}/{len(pdf_entries)} ({100*c/len(pdf_entries):.0f}%)")

    # Link to structured records
    records_df = pd.read_csv(args.records_csv)
    print(f"\nLoaded {len(records_df)} structured records")
    pdf_entries = link_records_to_pdfs(records_df, pdf_entries)

    # Save knowledge base
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pdf_entries, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved knowledge base ({len(pdf_entries)} entries) → {output_path}")

    # Summary stats
    with_records = sum(1 for e in pdf_entries if e.get("records"))
    total_records = sum(len(e.get("records", [])) for e in pdf_entries)
    avg_summary_len = sum(len(e["summary"]) for e in pdf_entries) / len(pdf_entries)
    print(f"\nSummary:")
    print(f"  PDFs with linked records: {with_records}")
    print(f"  Total linked records: {total_records}")
    print(f"  Average summary length: {avg_summary_len:.0f} chars")


if __name__ == "__main__":
    main()
