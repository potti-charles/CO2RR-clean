"""CO2RR Catalyst Knowledge Base Query System.

RAG-based Q&A system for CO2 electroreduction catalyst research.
Uses structured records + PDF literature as knowledge base,
DeepSeek API for reasoning and generation.

Usage:
    # Interactive mode
    python co2rr_knowledge_query.py

    # Single query
    python co2rr_knowledge_query.py --query "哪些Cu基催化剂生成C2H4效率最高？"

    # Catalyst recommendation
    python co2rr_knowledge_query.py --query "推荐一个CO2到HCOOH的高效催化剂" --mode recommend
"""

import argparse
from difflib import SequenceMatcher
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from runtime_config import OUTPUT_DIR, initialize_sample_data

initialize_sample_data()

KB_PATH = OUTPUT_DIR / "knowledge_base.json"
RECORDS_PATH = OUTPUT_DIR / "structured_records.csv"
DB_SUMMARY_PATH = OUTPUT_DIR / "db_summary_for_recommendation.md"
TFIDF_CACHE_PATH = OUTPUT_DIR / "tfidf_index.pkl"
FIGURE_INDEX_PATH = OUTPUT_DIR / "figure_index.json"
OCR_MARKDOWN_DIR = OUTPUT_DIR / "ocr_markdown"

DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_BASE_URL = "https://api.deepseek.com"
USER_MEMORY_PATH = OUTPUT_DIR / "user_memory.json"
MAX_HISTORY_TURNS = 4  # Keep last N Q&A pairs for context

SYSTEM_PROMPT = """你是一个 CO2 电催化还原（CO2RR）领域的专家知识库助手。

你的知识来源：
1. 当前加载的结构化数据库（学生版默认仅含少量示例），包含催化剂名称、产物、法拉第效率(FE%)、电位、电解质、合成方法等信息
2. 当前加载的 CO2RR 研究论文摘要和关键段落
3. 可选的论文表征图片（示例包不含原始论文图片）（XRD、SEM/TEM、XPS、电化学测试、法拉第效率图等），带有图片描述和分类

回答规则：
- 基于提供的数据和文献回答问题，不要编造数据
- 数值排名（如“最高”“最好”“TOP”）必须以结构化数据库检索结果为唯一依据；检索结果已按 FE 从高到低排序，不得用局部文献案例、用户画像或历史回答替换全局排名
- 用户画像只能调整回答表达方式，不能引入用户本次没有询问的产物或催化剂，也不能作为实验数据、最高值或推荐结论的证据
- 当前检索数据与历史回答冲突时，以当前检索数据为准，并明确纠正旧答案
- 查询多元素体系（如 Cu-Sn、CuSn）时，候选催化剂必须同时包含所有指定元素；不得返回只含其中一种元素的体系，也不得在无匹配时用全数据库高 FE 记录兜底
- 引用具体的催化剂名称、FE% 数值、实验条件
- **每个具体案例（催化剂、合成方法、实验数据）都必须标注来源DOI**，格式为 (DOI: 10.xxxx/xxxxx)，DOI必须从检索结果中获取，不要自己编造
- 如果某条信息在检索结果中找不到对应DOI，不要编造DOI，而是标注"（DOI未记录）"
- 如果数据库中没有相关信息，明确说明
- 回答简洁、有条理，用中文回答
- 涉及数据统计时，给出具体数字
- 涉及合成方法问题时，列出具体的合成方法及其对应的催化剂和性能数据
- 涉及表征问题时（XRD、电镜形貌、光谱、电化学测试等），引用相关图片的描述信息，说明典型特征并标注来源DOI
- 涉及电化学测试条件问题时，从数据中归纳常见的电位范围、电流密度范围
- 如果"用户画像"中有偏好信息，优先从该角度组织回答，但不要强行关联无关内容"""


# ---------------------------------------------------------------------------
# User memory / session profile
# ---------------------------------------------------------------------------

class UserMemory:
    """Lightweight persistent user memory — tracks interests and query history."""

    def __init__(self, path: Path = USER_MEMORY_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "query_history": [],      # last N queries (str)
            "interested_elements": {},  # element -> count
            "interested_products": {},  # product -> count
            "interested_topics": {},    # topic -> count  (synthesis, xrd, etc.)
            "noted_catalysts": [],      # catalysts user asked about
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_query(self, query: str, intent: dict):
        """Update memory from a new query + detected intent."""
        # Query history (keep last 30)
        hist = self.data["query_history"]
        hist.append(query)
        self.data["query_history"] = hist[-30:]

        # Accumulate element interests
        for elem in intent.get("elements", []):
            self.data["interested_elements"][elem] = self.data["interested_elements"].get(elem, 0) + 1

        # Accumulate product interests
        for prod in intent.get("products", []):
            self.data["interested_products"][prod] = self.data["interested_products"].get(prod, 0) + 1

        # Accumulate topic interests
        if intent.get("is_synthesis"):
            self.data["interested_topics"]["synthesis"] = self.data["interested_topics"].get("synthesis", 0) + 1
        for ct in intent.get("characterization_types", []):
            self.data["interested_topics"][ct] = self.data["interested_topics"].get(ct, 0) + 1
        if intent.get("is_recommendation"):
            self.data["interested_topics"]["recommendation"] = self.data["interested_topics"].get("recommendation", 0) + 1

        self.save()

    def get_profile_summary(self) -> str:
        """Build a concise user-profile string for injection into the system prompt."""
        parts = []
        elems = self.data.get("interested_elements", {})
        if elems:
            top_elems = sorted(elems.items(), key=lambda x: -x[1])[:5]
            parts.append(f"常关注元素: {', '.join(e for e, _ in top_elems)}")
        prods = self.data.get("interested_products", {})
        if prods:
            top_prods = sorted(prods.items(), key=lambda x: -x[1])[:5]
            parts.append(f"常关注产物: {', '.join(p for p, _ in top_prods)}")
        topics = self.data.get("interested_topics", {})
        if topics:
            top_topics = sorted(topics.items(), key=lambda x: -x[1])[:5]
            parts.append(f"常关注话题: {', '.join(t for t, _ in top_topics)}")
        recent = self.data.get("query_history", [])[-5:]
        if recent:
            parts.append(f"近期问题: {' / '.join(recent)}")
        if not parts:
            return ""
        return "\n".join(parts)


_user_memory = None

def get_user_memory() -> UserMemory:
    global _user_memory
    if _user_memory is None:
        _user_memory = UserMemory()
    return _user_memory


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_kb_cache = None
_records_cache = None


def load_knowledge_base() -> List[Dict]:
    global _kb_cache
    if _kb_cache is None:
        with open(KB_PATH, encoding="utf-8") as f:
            _kb_cache = json.load(f)
    return _kb_cache


def load_records() -> pd.DataFrame:
    global _records_cache
    if _records_cache is None:
        _records_cache = pd.read_csv(RECORDS_PATH)
    return _records_cache


# ---------------------------------------------------------------------------
# TF-IDF semantic index for literature retrieval
# ---------------------------------------------------------------------------

_tfidf_vectorizer = None
_tfidf_matrix = None


def _build_doc_text(entry: Dict) -> str:
    """Build a single document string from a KB entry for TF-IDF indexing."""
    parts = []
    sections = entry.get("sections", {})
    for key in ("abstract", "experimental", "results", "conclusion"):
        if sections.get(key):
            parts.append(sections[key])
    # Also add linked record info
    for rec in entry.get("records", []):
        parts.append(
            f"{rec.get('catalyst_name','')} {rec.get('product','')} "
            f"FE {rec.get('fe_pct','')}%"
        )
    return " ".join(parts)


def build_tfidf_index(force: bool = False):
    """Build or load TF-IDF index over the knowledge base."""
    global _tfidf_vectorizer, _tfidf_matrix

    if _tfidf_vectorizer is not None and not force:
        return

    # Try loading cached index
    if TFIDF_CACHE_PATH.exists() and not force:
        with open(TFIDF_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
            _tfidf_vectorizer = cached["vectorizer"]
            _tfidf_matrix = cached["matrix"]
        return

    kb = load_knowledge_base()
    docs = [_build_doc_text(entry) for entry in kb]

    _tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(docs)

    # Cache
    TFIDF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFIDF_CACHE_PATH, "wb") as f:
        pickle.dump({"vectorizer": _tfidf_vectorizer, "matrix": _tfidf_matrix}, f)


def retrieve_literature_semantic(query: str, top_k: int = 5) -> List[Dict]:
    """Retrieve literature using TF-IDF semantic similarity."""
    build_tfidf_index()
    kb = load_knowledge_base()

    query_vec = _tfidf_vectorizer.transform([query])
    scores = sklearn_cosine(query_vec, _tfidf_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0.01:  # minimal threshold
            results.append(kb[idx])
    return results


# ---------------------------------------------------------------------------
# Single-paper retrieval and Q&A
# ---------------------------------------------------------------------------

SINGLE_PAPER_SYSTEM_PROMPT = """你是一个 CO2RR 单篇论文深度阅读助手。

回答要求：
- 主要依据“当前锁定论文”的证据片段回答，不得把其他论文的数据归给当前论文
- 区分法拉第效率、总 C2/C2+ FE、总 CO2RR FE、选择性、质量百分比和产率
- 涉及实验条件或数值时，必须在句末标注证据位置，格式为 [experimental #2]、[results #4]
- 如果给定证据不足，明确说“当前论文证据中未找到”，不要推测
- 图表问题只能根据图注和检索到的正文解释；没有图注或图像信息时明确说明
- 只有用户明确要求比较时，才使用“对比论文”材料，并清楚区分来源 DOI
- 不输出或逐字复现论文全文；提供有证据的分析、摘要和必要短摘录
- 用中文回答，先给结论，再说明证据和局限性
"""


def _normalize_doi(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^\s*doi\s*:\s*", "", value)
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    match = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", value, re.IGNORECASE)
    if match:
        value = match.group(0)
    value = value.rstrip("/.,;:)]}> ")
    return re.sub(r"(?:/-)?/dcsupplemental$", "", value, flags=re.IGNORECASE)


_single_paper_corpus = None


def _entry_from_ocr_markdown(path: Path) -> Dict:
    """Build a section-aware paper entry from one OCR Markdown file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", raw, re.IGNORECASE)
    doi = doi_match.group(0).rstrip(".,;) ") if doi_match else ""

    lines = raw.splitlines()
    title = ""
    for line in lines[:30]:
        stripped = line.strip().lstrip("#").strip()
        if not stripped or stripped.startswith("<!--"):
            continue
        if stripped.lower() == path.stem.lower():
            continue
        title = stripped
        break
    if not title:
        title = path.stem.replace("-", " ")

    heading_pattern = re.compile(
        r"(?im)^\*\*(ABSTRACT|INTRODUCTION|RESULTS(?: AND DISCUSSION)?|"
        r"EXPERIMENTAL(?: SECTION)?|METHODS?|CONCLUSIONS?|LIMITATIONS?)\s*:?\*\*\s*"
    )
    matches = list(heading_pattern.finditer(raw))
    sections = {}
    section_names = {
        "abstract": "abstract",
        "introduction": "introduction",
        "results": "results",
        "results and discussion": "results",
        "experimental": "experimental",
        "experimental section": "experimental",
        "method": "experimental",
        "methods": "experimental",
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "limitation": "conclusion",
        "limitations": "conclusion",
    }
    if matches:
        for idx, match in enumerate(matches):
            key = section_names.get(match.group(1).lower(), match.group(1).lower())
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
            text = raw[match.end():end].strip()
            sections[key] = (sections.get(key, "") + "\n" + text).strip()
    else:
        sections["full_text"] = raw

    return {
        "title": title,
        "doi": doi,
        "pdf_file": path.with_suffix(".pdf").name,
        "ocr_markdown_file": str(path),
        "full_text_chars": len(raw),
        "sections": sections,
        "records": [],
    }


def load_single_paper_corpus() -> List[Dict]:
    """Combine full text, OCR, and structured-only papers into one index."""
    global _single_paper_corpus
    if _single_paper_corpus is not None:
        return _single_paper_corpus

    corpus = list(load_knowledge_base())
    known_dois = {
        _normalize_doi(str(entry.get("doi", "")))
        for entry in corpus if entry.get("doi")
    }
    stem_to_index = {
        Path(str(entry.get("pdf_file", ""))).stem.lower(): index
        for index, entry in enumerate(corpus) if entry.get("pdf_file")
    }
    if OCR_MARKDOWN_DIR.exists():
        for path in sorted(OCR_MARKDOWN_DIR.glob("*.md")):
            entry = _entry_from_ocr_markdown(path)
            doi = _normalize_doi(str(entry.get("doi", "")))
            existing_index = stem_to_index.get(path.stem.lower())
            if existing_index is not None:
                existing_doi = _normalize_doi(
                    str(corpus[existing_index].get("doi", ""))
                )
                if doi and doi != existing_doi:
                    # The OCR belongs to the same local PDF and is therefore
                    # authoritative when the legacy KB linked that PDF to a
                    # different DOI.
                    corpus[existing_index] = entry
                    if existing_doi:
                        known_dois.discard(existing_doi)
                    known_dois.add(doi)
                continue
            if doi and doi in known_dois:
                continue
            corpus.append(entry)
            stem_to_index[path.stem.lower()] = len(corpus) - 1
            if doi:
                known_dois.add(doi)

    # Every structured paper must remain lockable even when its local PDF/OCR
    # is unavailable.  Such entries are explicitly marked structured_only so
    # the UI and answer generator do not imply that full text was consulted.
    records = load_records()
    if "doi" in records.columns:
        normalized = records["doi"].astype(str).map(_normalize_doi)
        for doi, group in records.assign(_normalized_doi=normalized).groupby(
            "_normalized_doi", sort=False
        ):
            if (
                not re.fullmatch(r"10\.\d{4,9}/\S+", doi)
                or doi in known_dois
            ):
                continue
            first = group.iloc[0]
            title = str(first.get("title", "") or "").strip()
            measurements = []
            linked_records = []
            for _, row in group.iterrows():
                product = str(row.get("product", "") or "")
                fe = row.get("fe_pct", "")
                voltage = row.get("voltage_V", "")
                electrolyte = str(row.get("electrolyte", "") or "")
                measurements.append(
                    f"{product}: FE {fe}% at {voltage} V; electrolyte {electrolyte}."
                )
                linked_records.append({
                    key: row.get(key, "")
                    for key in (
                        "catalyst_name", "catalyst_type", "product", "fe_pct",
                        "voltage_V", "electrolyte", "cell_setup",
                        "current_density_mA_cm2", "synthesis_method",
                        "product_rank", "is_representative", "evidence_text",
                    ) if key in group.columns
                })
            summary = (
                f"Structured database record only. Title: {title}. DOI: {doi}. "
                + " ".join(measurements)
            )
            corpus.append({
                "pdf_file": "", "title": title, "full_text_chars": 0,
                "sections": {"structured_data": summary}, "summary": summary,
                "records": linked_records, "doi": doi,
                "doi_source": "structured_database",
                "doi_confidence": "curated",
                "source_level": "structured_only",
            })
            known_dois.add(doi)

    _single_paper_corpus = corpus
    return corpus


def search_papers(query: str, top_k: int = 5) -> List[Dict]:
    """Find papers by exact DOI, title substring, or fuzzy title similarity."""
    needle = (query or "").strip()
    if not needle:
        return []

    needle_lower = needle.lower()
    needle_doi = _normalize_doi(needle)
    looks_like_doi = bool(re.fullmatch(r"10\.\d{4,9}/\S+", needle_doi))
    scored = []
    for index, entry in enumerate(load_single_paper_corpus()):
        doi = _normalize_doi(str(entry.get("doi", "")))
        sections = entry.get("sections", {})
        title = _paper_title(entry)
        abstract = str(sections.get("abstract", "") or "").strip()
        title_lower = title.lower()
        pdf_lower = str(entry.get("pdf_file", "")).lower()
        searchable_text = f"{title_lower} {pdf_lower} {abstract[:3000].lower()}"

        if doi and needle_doi == doi:
            score = 100.0
        elif looks_like_doi:
            continue
        elif needle_lower in searchable_text:
            score = 80.0 + min(len(needle_lower) / max(len(title_lower), 1), 1.0)
        else:
            score = SequenceMatcher(None, needle_lower, searchable_text[:500]).ratio() * 50.0
            query_tokens = set(re.findall(r"[a-z0-9]+", needle_lower))
            title_tokens = set(re.findall(r"[a-z0-9]+", title_lower))
            if query_tokens:
                score += 20.0 * len(query_tokens & title_tokens) / len(query_tokens)

        if score >= 12.0:
            scored.append({
                "index": index,
                "score": score,
                "doi": entry.get("doi", ""),
                "title": title,
                "pdf_file": entry.get("pdf_file", ""),
                "source_level": entry.get("source_level", "full_text"),
            })

    scored.sort(key=lambda item: (-item["score"], item["title"]))
    return scored[:top_k]


def get_paper(index: int) -> Dict:
    """Return one knowledge-base entry by its stable in-file index."""
    kb = load_single_paper_corpus()
    if index is None or int(index) < 0 or int(index) >= len(kb):
        raise IndexError("论文索引无效")
    return kb[int(index)]


def _paper_title(entry: Dict) -> str:
    title = str(entry.get("title", "") or "").strip()
    if title:
        return title
    doi = _normalize_doi(str(entry.get("doi", "")))
    if doi:
        records = load_records()
        matches = records[
            records["doi"].astype(str).str.lower().map(_normalize_doi).eq(doi)
        ]
        if not matches.empty and "title" in matches.columns:
            structured_title = str(matches.iloc[0]["title"] or "").strip()
            if structured_title and structured_title.lower() != "nan":
                return structured_title
    abstract = str(entry.get("sections", {}).get("abstract", "") or "").strip()
    pdf_file = str(entry.get("pdf_file", "") or "").strip()
    return pdf_file or (abstract.splitlines()[0][:240] if abstract else "未知论文")


def _chunk_paper(entry: Dict, chunk_size: int = 1800, overlap: int = 250) -> List[Dict]:
    """Split the selected paper into section-aware, overlapping character chunks."""
    chunks = []
    for section, raw_text in entry.get("sections", {}).items():
        text = re.sub(r"\s+", " ", str(raw_text or "")).strip()
        if not text:
            continue
        start = 0
        number = 1
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(". ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary + 1
            chunks.append({
                "section": section,
                "number": number,
                "start": start,
                "end": end,
                "text": text[start:end],
            })
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
            number += 1
    return chunks


def retrieve_paper_chunks(entry: Dict, query: str, top_k: int = 7) -> List[Dict]:
    """Retrieve the most relevant chunks from one selected paper."""
    chunks = _chunk_paper(entry)
    if not chunks:
        return []

    docs = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(docs + [query])
    scores = sklearn_cosine(matrix[-1], matrix[:-1])[0]

    q_lower = query.lower()
    section_boosts = {}
    if any(key in q_lower for key in ("实验", "方法", "合成", "制备", "条件", "method", "synthesis")):
        section_boosts["experimental"] = 0.35
    if any(key in q_lower for key in ("机理", "机制", "为什么", "mechanism")):
        section_boosts["results"] = 0.25
    if any(key in q_lower for key in ("结论", "局限", "不足", "conclusion", "limitation")):
        section_boosts["conclusion"] = 0.35
    if any(key in q_lower for key in ("图", "表", "figure", "table", "xrd", "sem", "tem", "xps")):
        section_boosts["results"] = 0.2

    ranked = []
    for idx, chunk in enumerate(chunks):
        score = float(scores[idx]) + section_boosts.get(chunk["section"], 0.0)
        ranked.append((score, idx, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in ranked[:top_k]]


def _single_paper_figure_context(entry: Dict, query: str, top_k: int = 8) -> str:
    doi = _normalize_doi(str(entry.get("doi", "")))
    pdf_name = Path(str(entry.get("pdf_file", ""))).name.lower()
    if not doi and not pdf_name:
        return ""
    figures = [
        figure for figure in load_figure_index()
        if (
            doi and _normalize_doi(str(figure.get("doi", ""))) == doi
        ) or (
            pdf_name and Path(str(figure.get("pdf", ""))).name.lower() == pdf_name
        )
    ]
    if not figures:
        return ""

    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    figures.sort(
        key=lambda figure: -sum(
            term in str(figure.get("caption", "")).lower() for term in terms
        )
    )
    lines = ["### 当前论文图表索引"]
    for figure in figures[:top_k]:
        label = figure.get("label") or figure.get("file") or "figure"
        caption = re.sub(r"\s+", " ", str(figure.get("caption", ""))).strip()
        lines.append(f"- {label}: {caption[:700]}")
    return "\n".join(lines)


def build_single_paper_context(entry: Dict, query: str) -> str:
    """Build grounded context for a question about one locked paper."""
    chunks = retrieve_paper_chunks(entry, query)
    lines = [
        "### 当前锁定论文",
        f"- 标题: {_paper_title(entry)}",
        f"- DOI: {entry.get('doi') or '未记录'}",
        f"- PDF/记录: {entry.get('pdf_file') or '未记录'}",
    ]
    if entry.get("source_level") == "structured_only":
        lines.extend([
            "- 证据级别: 仅结构化数据库（无本地全文/OCR）",
            "- 回答限制: 只能回答下列结构化实验字段；不得推断机理、图表、结论或局限性。",
        ])
    lines.append("\n### 当前论文相关证据片段")
    if chunks:
        for chunk in chunks:
            label = f"{chunk['section']} #{chunk['number']}"
            lines.append(f"\n[{label}]\n{chunk['text']}")
    else:
        lines.append("当前知识库没有可用全文片段。")

    figure_context = _single_paper_figure_context(entry, query)
    if figure_context:
        lines.append("\n" + figure_context)

    if any(keyword in query.lower() for keyword in ("对比", "比较", "区别", "差异", "compare", "versus", " vs")):
        intent = detect_query_intent(query)
        comparisons = [
            paper for paper in retrieve_literature(query, intent, top_k=5)
            if _normalize_doi(str(paper.get("doi", "")))
            != _normalize_doi(str(entry.get("doi", "")))
        ][:3]
        if comparisons:
            lines.append("\n### 对比论文（仅用于明确的比较问题）")
            for paper in comparisons:
                abstract = re.sub(
                    r"\s+", " ", str(paper.get("sections", {}).get("abstract", ""))
                ).strip()
                lines.append(
                    f"- DOI: {paper.get('doi') or '未记录'} | "
                    f"{_paper_title(paper)}\n  摘要证据: {abstract[:900]}"
                )
    return "\n".join(lines)


def answer_single_paper_stream(
    query: str,
    paper_index: int,
    client=None,
    model: str = DEFAULT_MODEL,
    chat_history: list = None,
):
    """Stream a grounded answer about one selected paper."""
    entry = get_paper(paper_index)
    context = build_single_paper_context(entry, query)
    prompt = f"""用户正在深入阅读一篇已经锁定的论文。

用户问题：{query}

{context}

请严格依据上述证据回答，并标注证据位置。"""

    llm_history = []
    if chat_history:
        for user_q, assistant_a in chat_history[-MAX_HISTORY_TURNS:]:
            llm_history.append({"role": "user", "content": user_q})
            llm_history.append({"role": "assistant", "content": assistant_a})

    if client is None:
        client = get_client()

    partial = ""
    for chunk in query_llm_stream(
        client,
        SINGLE_PAPER_SYSTEM_PROMPT,
        prompt,
        model=model,
        chat_history=llm_history,
    ):
        partial += chunk
        yield partial, context


# ---------------------------------------------------------------------------
# Keyword-based retrieval (original, used as fallback / boost)
# ---------------------------------------------------------------------------

# Keywords for different product categories
PRODUCT_KEYWORDS = {
    "CO": ["co ", "carbon monoxide", "co2-to-co", "co2 to co", "一氧化碳"],
    "HCOOH": ["formate", "formic acid", "hcooh", "hcoo", "甲酸", "formate"],
    "C2H4": ["ethylene", "c2h4", "c2 product", "c2+", "乙烯"],
    "C2H5OH": ["ethanol", "c2h5oh", "乙醇"],
    "CH4": ["methane", "ch4", "methanation", "甲烷"],
    "CH3OH": ["methanol", "ch3oh", "甲醇"],
}

C2_PRODUCTS = ["C2H4", "C2H5OH", "C2H6", "CH3CHO", "CH3COOH"]
OD_CU_PATTERN = re.compile(
    r"oxide[- /]?derived|oxide-/hydroxide-derived|cuo[- ]derived|"
    r"cu2o[- ]derived|oxidized copper|od[- ]?cu",
    re.IGNORECASE,
)
PLAIN_CU_PATTERN = re.compile(
    r"^(?:metallic cu|polycrystalline (?:cu|copper)|(?:cu|copper) foil|"
    r"cu|copper)$",
    re.IGNORECASE,
)

CATALYST_KEYWORDS = {
    "Cu": ["copper", " cu ", "cu-", "cu2o", "cuo", "cu基", "铜基"],
    "Sn": ["tin", " sn ", "sn-", "cu-sn", "cusn", "sn基", "锡基"],
    "Ag": ["silver", " ag ", "ag-", "cu-ag", "cuag"],
    "Bi": ["bismuth", " bi ", "bi-", "cu-bi"],
    "In": ["indium", " in ", "in-", "cu-in", "cuin"],
    "Au": ["gold", " au ", "au-", "cu-au"],
    "Zn": ["zinc", " zn ", "zn-", "cu-zn"],
    "Fe": ["iron", " fe ", "fe-", "cu-fe"],
    "Co": ["cobalt", " co-", "co-cu"],  # careful: "co" overlaps with CO product
    "Ni": ["nickel", " ni ", "ni-", "cu-ni"],
    "Pd": ["palladium", " pd ", "pd-"],
}

ELEMENT_NAME_ALIASES = {
    "Cu": ["copper"],
    "Sn": ["tin"],
    "Ag": ["silver"],
    "Bi": ["bismuth"],
    "In": ["indium"],
    "Au": ["gold"],
    "Zn": ["zinc"],
    "Fe": ["iron"],
    "Co": ["cobalt"],
    "Ni": ["nickel"],
    "Pd": ["palladium"],
}


def _catalyst_matches_element(catalyst_name: str, element: str) -> bool:
    """Match an element in a catalyst name without substring false positives."""
    name = str(catalyst_name or "")
    symbols = set(re.findall(r"[A-Z][a-z]?", name))
    if element in symbols:
        return True
    lower_name = name.lower()
    return any(
        re.search(rf"\b{re.escape(alias)}\b", lower_name)
        for alias in ELEMENT_NAME_ALIASES.get(element, [])
    )

STRUCTURE_KEYWORDS = {
    "nanoparticle": ["nanoparticle", "np ", "nps ", "纳米颗粒", "纳米粒子"],
    "nanowire": ["nanowire", "nw ", "nws ", "纳米线"],
    "nanocube": ["nanocube", "nc ", "纳米立方体"],
    "nanosheet": ["nanosheet", "ns ", "纳米片"],
    "foam": ["foam", "porous", "泡沫", "多孔"],
    "dendrite": ["dendrit", "树枝状"],
    "oxide-derived": ["oxide-derived", "od-cu", "oxide derived", "氧化物衍生"],
    "single-atom": ["single-atom", "single atom", "sac", "atomically dispersed", "单原子"],
    "MOF": ["mof", "metal-organic framework", "metal organic framework", "金属有机框架"],
    "alloy": ["alloy", "bimetal", "intermetallic", "合金"],
}


def detect_query_intent(query: str) -> Dict:
    """Detect what the user is asking about."""
    q_lower = query.lower()
    intent = {
        "products": [],
        "elements": [],
        "structures": [],
        "is_recommendation": False,
        "is_comparison": False,
        "is_trend": False,
        "is_top_n": False,
        "is_c2_family": False,
        "compare_plain_cu": False,
    }

    # Detect products
    for product, keywords in PRODUCT_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            intent["products"].append(product)

    c2_family_keywords = [
        "c2产物", "c₂产物", "c2 产品", "c₂ 产品", "c2 products",
        "c2 product", "c2+", "c₂+",
    ]
    intent["is_c2_family"] = any(kw in q_lower for kw in c2_family_keywords)
    if intent["is_c2_family"]:
        intent["products"] = list(C2_PRODUCTS)

    # Detect elements
    for elem, keywords in CATALYST_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            intent["elements"].append(elem)

    # Compact bimetallic spellings such as CuSn do not contain the delimiter
    # expected by the individual keyword lists.  Detect common Cu-M pairs as
    # one chemical composition and require both elements downstream.
    pair_aliases = {
        "Sn": (r"cu\s*[-/]?\s*sn", "铜锡"),
        "Ag": (r"cu\s*[-/]?\s*ag", "铜银"),
        "Bi": (r"cu\s*[-/]?\s*bi", "铜铋"),
        "In": (r"cu\s*[-/]?\s*in", "铜铟"),
        "Au": (r"cu\s*[-/]?\s*au", "铜金"),
        "Zn": (r"cu\s*[-/]?\s*zn", "铜锌"),
        "Pd": (r"cu\s*[-/]?\s*pd", "铜钯"),
    }
    for second_element, aliases in pair_aliases.items():
        if re.search(aliases[0], q_lower) or aliases[1] in query:
            if "Cu" not in intent["elements"]:
                intent["elements"].append("Cu")
            if second_element not in intent["elements"]:
                intent["elements"].append(second_element)

    # Detect structures
    for struct, keywords in STRUCTURE_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            intent["structures"].append(struct)

    # Detect query type
    recommend_kw = ["推荐", "建议", "设计", "recommend", "suggest", "design", "propose"]
    compare_kw = ["对比", "比较", "区别", "差异", "vs", "compare", "difference"]
    trend_kw = ["趋势", "影响", "效应", "关系", "trend", "effect", "influence", "relationship"]
    top_kw = ["最好", "最高", "top", "best", "highest", "排名", "leaderboard"]
    synthesis_kw = [
        "合成", "制备", "合成方法", "制备方法", "怎么做", "如何合成", "制造",
        "synthesis", "preparation", "fabricat", "synthesize", "prepare",
        "electrodeposition", "电沉积", "水热", "溶剂热", "hydrothermal",
        "solvothermal", "calcination", "煅烧", "退火", "annealing",
        "sputtering", "溅射", "cvd", "pvd", "sol-gel", "溶胶凝胶",
        "共沉淀", "co-precipitation", "热处理", "pyrolysis", "热解",
    ]

    intent["is_recommendation"] = any(kw in q_lower for kw in recommend_kw)
    intent["is_comparison"] = any(kw in q_lower for kw in compare_kw)
    intent["is_trend"] = any(kw in q_lower for kw in trend_kw)
    intent["is_top_n"] = any(kw in q_lower for kw in top_kw)
    intent["is_synthesis"] = any(kw in q_lower for kw in synthesis_kw)
    plain_cu_keywords = [
        "普通cu", "普通 cu", "普通铜", "多晶cu", "多晶 cu", "多晶铜",
        "metallic cu", "polycrystalline cu", "pristine cu", "bare cu",
        "untreated cu", "conventional cu",
    ]
    intent["compare_plain_cu"] = (
        intent["is_comparison"]
        and "oxide-derived" in intent["structures"]
        and any(keyword in q_lower for keyword in plain_cu_keywords)
    )

    # Detect characterization / figure-related queries
    char_keywords = {
        "xrd": ["xrd", "x射线衍射", "衍射", "diffraction", "分峰", "晶相", "晶体结构",
                 "crystal structure", "crystal phase", "2θ", "peaks"],
        "electron_microscopy": ["sem", "tem", "电镜", "电子显微镜", "形貌", "morpholog",
                                "微观结构", "nanostructure", "haadf", "hrtem", "edx",
                                "eds", "saed", "electron microscop"],
        "spectroscopy": ["xps", "光谱", "raman", "ftir", "红外", "拉曼", "光电子能谱",
                         "xanes", "exafs", "uv-vis", "auger", "spectroscop"],
        "faradaic_efficiency": ["法拉第效率", "faradaic", "fe%", "选择性", "selectiv",
                                "产物分布", "product distribution"],
        "current_density": ["电流密度", "current density", "极化曲线", "j-v"],
        "stability": ["稳定性", "stability", "寿命", "durability", "长期"],
        "electrochemistry": ["cv", "lsv", "循环伏安", "线性扫描", "阻抗", "eis",
                             "nyquist", "tafel", "电化学测试", "电位窗口",
                             "electrochemical test", "potential window"],
    }
    intent["characterization_types"] = []
    for char_type, kws in char_keywords.items():
        if any(kw in q_lower for kw in kws):
            intent["characterization_types"].append(char_type)

    return intent


def retrieve_records(query: str, intent: Dict, top_k: int = 20) -> pd.DataFrame:
    """Retrieve relevant structured records based on query intent."""
    df = load_records()

    if intent.get("compare_plain_cu"):
        product_mask = pd.Series([True] * len(df), index=df.index)
        if intent["products"]:
            product_mask = df["product"].isin(intent["products"])
        names = df["catalyst_name"].fillna("")
        cu_mask = names.map(lambda name: _catalyst_matches_element(name, "Cu"))
        od_rows = df[
            product_mask & cu_mask & names.str.contains(OD_CU_PATTERN)
        ].copy()
        plain_rows = df[
            product_mask & cu_mask & names.str.match(PLAIN_CU_PATTERN)
        ].copy()
        od_rows["comparison_group"] = "oxide-derived Cu"
        plain_rows["comparison_group"] = "普通/多晶 Cu"
        per_group = max(1, top_k // 2)
        return pd.concat(
            [
                od_rows.nlargest(per_group, "fe_pct"),
                plain_rows.nlargest(per_group, "fe_pct"),
            ],
            ignore_index=True,
        )

    mask = pd.Series([True] * len(df))

    # Filter by product
    if intent["products"]:
        mask = mask & df["product"].isin(intent["products"])

    # Filter by element. Multi-element queries use AND semantics: Cu-Sn must
    # contain both Cu and Sn, never merely one of them.
    if intent["elements"]:
        elem_mask = pd.Series([True] * len(df), index=df.index)
        for elem in intent["elements"]:
            current_element_mask = df["catalyst_name"].map(
                lambda name: _catalyst_matches_element(name, elem)
            )
            elem_mask = elem_mask & current_element_mask
        mask = mask & elem_mask

    # Filter by structure
    if intent["structures"]:
        struct_mask = pd.Series([False] * len(df))
        for struct in intent["structures"]:
            keywords = STRUCTURE_KEYWORDS.get(struct, [struct])
            for kw in keywords:
                struct_mask = struct_mask | df["catalyst_name"].str.contains(
                    kw, case=False, na=False
                )
                if "catalyst_type" in df.columns:
                    struct_mask = struct_mask | df["catalyst_type"].str.contains(
                        kw, case=False, na=False
                    )
        mask = mask & struct_mask

    filtered = df[mask]

    # Sort by FE% descending and return top_k
    return filtered.nlargest(top_k, "fe_pct")


def retrieve_literature(query: str, intent: Dict, top_k: int = 5) -> List[Dict]:
    """Retrieve relevant PDF literature using hybrid: TF-IDF semantic + keyword boost."""
    # Semantic retrieval (wider pool)
    semantic_results = retrieve_literature_semantic(query, top_k=top_k * 3)

    # Score with keyword boosts
    q_lower = query.lower()
    scored = []
    for entry in semantic_results:
        if len(intent["elements"]) > 1:
            record_names = [
                str(record.get("catalyst_name", ""))
                for record in entry.get("records", [])
            ]
            if not record_names or not any(
                all(
                    _catalyst_matches_element(name, element)
                    for element in intent["elements"]
                )
                for name in record_names
            ):
                continue
        score = 1.0  # base score from semantic match

        # Boost if has linked records matching intent
        for rec in entry.get("records", []):
            if intent["products"] and rec.get("product") in intent["products"]:
                score += 3
            if intent["elements"]:
                for elem in intent["elements"]:
                    if elem in rec.get("catalyst_name", ""):
                        score += 2

        # Boost by section availability
        if entry.get("sections", {}).get("conclusion"):
            score += 0.5

        scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:top_k]]


def retrieve_comparison_literature(records: pd.DataFrame, per_group: int = 3) -> List[Dict]:
    """Retrieve DOI-linked papers evenly from each explicit comparison group."""
    if "comparison_group" not in records.columns:
        return []
    by_doi = {
        _normalize_doi(str(entry.get("doi", ""))): entry
        for entry in load_knowledge_base()
        if entry.get("doi")
    }
    results = []
    for group, group_rows in records.groupby("comparison_group", sort=False):
        added = 0
        for _, row in group_rows.iterrows():
            entry = by_doi.get(_normalize_doi(str(row.get("doi", ""))))
            if not entry:
                continue
            labelled_entry = dict(entry)
            labelled_entry["comparison_group"] = group
            results.append(labelled_entry)
            added += 1
            if added >= per_group:
                break
    return results


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def build_data_context(records: pd.DataFrame) -> str:
    """Build data context string from retrieved records."""
    if records.empty:
        return "（数据库中没有找到匹配的记录）"

    lines = [f"### 数据库检索结果（共 {len(records)} 条相关记录）\n"]

    if "comparison_group" in records.columns:
        lines.append("- 以下记录来自两个独立检索组，必须分别比较，不得用一组推测另一组。")
        for group, group_rows in records.groupby("comparison_group", sort=False):
            lines.append(
                f"- {group}: {len(group_rows)} 条，FE 范围 "
                f"{group_rows['fe_pct'].min():.1f}%–{group_rows['fe_pct'].max():.1f}%"
            )

    # Summary stats
    lines.append(f"- FE% 范围: {records['fe_pct'].min():.1f}% - {records['fe_pct'].max():.1f}%")
    lines.append(f"- FE% 均值: {records['fe_pct'].mean():.1f}%, 中位数: {records['fe_pct'].median():.1f}%")

    if "product" in records.columns:
        product_stats = records.groupby("product")["fe_pct"].agg(["count", "mean", "max"])
        lines.append(f"\n按产物分布:")
        for product, row in product_stats.iterrows():
            lines.append(f"  - {product}: {int(row['count'])}条, 均值FE={row['mean']:.1f}%, 最高={row['max']:.1f}%")

    # Synthesis method stats (if column exists and has data)
    if "synthesis_method" in records.columns:
        syn_records = records[records["synthesis_method"].notna() & (records["synthesis_method"] != "")]
        if len(syn_records) > 0:
            from collections import Counter, defaultdict
            all_methods = []
            method_dois = defaultdict(list)
            for _, r in syn_records.iterrows():
                for m in r["synthesis_method"].split("; "):
                    m = m.strip()
                    if m:
                        all_methods.append(m)
                        doi = r.get("doi", "")
                        cat = r.get("catalyst_name", "")
                        if pd.notna(doi) and doi:
                            method_dois[m].append((cat, doi))
            method_counts = Counter(all_methods)
            lines.append(f"\n合成方法分布 ({len(syn_records)}/{len(records)} 条有合成方法数据):")
            for method, count in method_counts.most_common(8):
                examples = method_dois.get(method, [])[:2]
                doi_str = "; ".join(f"{cat} (DOI: {doi})" for cat, doi in examples)
                if doi_str:
                    lines.append(f"  - {method}: {count}条, 例如: {doi_str}")
                else:
                    lines.append(f"  - {method}: {count}条")

    # Top records
    top = records if "comparison_group" in records.columns else records.head(10)
    lines.append(f"\n前 {len(top)} 条最佳记录:")
    for _, row in top.iterrows():
        parts = [f"  - {row['catalyst_name']}"]
        if "comparison_group" in records.columns:
            parts.append(f"[对照组: {row.get('comparison_group', '')}]")
        if pd.notna(row.get("catalyst_type")) and row["catalyst_type"]:
            parts.append(f"({row['catalyst_type']})")
        parts.append(f"→ {row['product']}: FE={row['fe_pct']:.1f}%")
        if pd.notna(row.get("voltage_V")):
            parts.append(f"at {row['voltage_V']}V")
        if pd.notna(row.get("electrolyte")) and row["electrolyte"]:
            parts.append(f"in {row['electrolyte']}")
        if pd.notna(row.get("synthesis_method")) and row["synthesis_method"]:
            parts.append(f"[合成: {row['synthesis_method']}]")
        if pd.notna(row.get("doi")):
            parts.append(f"(DOI: {row['doi']})")
        lines.append(" ".join(parts))

    return "\n".join(lines)


def build_ranking_guard(intent: Dict) -> str:
    """Return a hard constraint for deterministic numeric leaderboard answers."""
    if not intent.get("is_top_n"):
        return ""
    return """
### 排名回答强约束
这是全库数值排名查询。上方“前 N 条最佳记录”已经按 FE 从高到低排列。
- 直接按该顺序回答，不得从相关文献、用户画像或历史对话另选案例。
- 不得把局部召回中的最高值表述为全库最高值。
- 不得复用历史回答里的数值；若历史回答与当前记录冲突，以当前记录为准。
"""


def build_literature_context(lit_entries: List[Dict]) -> str:
    """Build literature context string from retrieved PDF entries."""
    if not lit_entries:
        return ""

    lines = [f"\n### 相关文献摘要（{len(lit_entries)} 篇）\n"]

    for entry in lit_entries:
        lines.append(f"**{entry['pdf_file']}**")
        if entry.get("comparison_group"):
            lines.append(f"对照组: {entry['comparison_group']}")
        if entry.get("doi"):
            lines.append(f"DOI: {entry['doi']}")

        # Show abstract (truncated)
        abstract = entry.get("sections", {}).get("abstract", "")
        if abstract:
            lines.append(f"摘要: {abstract[:500]}...")

        # Show conclusion if available
        conclusion = entry.get("sections", {}).get("conclusion", "")
        if conclusion:
            lines.append(f"结论: {conclusion[:400]}...")

        # Show linked records
        if entry.get("records"):
            lines.append("实验数据:")
            for rec in entry["records"][:5]:
                v_str = f" at {rec['voltage_V']}V" if rec.get("voltage_V") else ""
                lines.append(f"  - {rec['catalyst_name']} → {rec['product']}: FE={rec['fe_pct']}%{v_str}")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure / characterization retrieval
# ---------------------------------------------------------------------------

_figure_index_cache = None


def load_figure_index() -> List[Dict]:
    """Load the figure index JSON."""
    global _figure_index_cache
    if _figure_index_cache is None:
        if FIGURE_INDEX_PATH.exists():
            with open(FIGURE_INDEX_PATH, encoding="utf-8") as f:
                _figure_index_cache = json.load(f)
        else:
            _figure_index_cache = []
    return _figure_index_cache


def retrieve_figures(intent: Dict, top_k: int = 10) -> List[Dict]:
    """Retrieve relevant figures based on query intent."""
    fig_index = load_figure_index()
    if not fig_index:
        return []

    char_types = intent.get("characterization_types", [])
    elements = intent.get("elements", [])
    structures = intent.get("structures", [])

    scored = []
    for fig in fig_index:
        score = 0
        category = fig.get("category", "other")
        caption = (fig.get("caption", "") or "").lower()
        pdf_name = fig.get("pdf", "").lower()

        # Match by characterization type
        if char_types:
            if category in char_types:
                score += 5
            # Also check caption text
            for ct in char_types:
                if ct in caption:
                    score += 2

        # Match by element
        for elem in elements:
            if elem.lower() in caption or elem.lower() in pdf_name:
                score += 3

        # Match by structure
        for struct in structures:
            kws = STRUCTURE_KEYWORDS.get(struct, [struct])
            for kw in kws:
                if kw.lower() in caption:
                    score += 2

        # Skip uncategorized figures unless they score on other criteria
        if category == "other" and score < 3:
            continue

        # Skip figures with no caption and low score
        if not fig.get("caption") and score < 5:
            continue

        if score > 0:
            scored.append((score, fig))

    scored.sort(key=lambda x: -x[0])
    return [fig for _, fig in scored[:top_k]]


def build_figure_context(figures: List[Dict]) -> str:
    """Build figure context string for LLM."""
    if not figures:
        return ""

    lines = [f"\n### 相关表征图片信息（{len(figures)} 张）\n"]

    # Group by category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for fig in figures:
        by_cat[fig.get("category", "other")].append(fig)

    cat_names = {
        "xrd": "XRD 衍射图谱",
        "electron_microscopy": "电子显微镜 (SEM/TEM)",
        "spectroscopy": "光谱学 (XPS/Raman/FTIR等)",
        "electrochemistry": "电化学测试 (CV/LSV/EIS等)",
        "faradaic_efficiency": "法拉第效率",
        "current_density": "电流密度",
        "stability": "稳定性测试",
        "morphology_scheme": "结构示意图",
        "other": "其他",
    }

    for cat, cat_figs in by_cat.items():
        lines.append(f"\n**{cat_names.get(cat, cat)}:**")
        for fig in cat_figs[:5]:
            caption = fig.get("caption", "")[:200]
            doi = fig.get("doi", "")
            doi_str = f" (DOI: {doi})" if doi else ""
            fig_file = fig.get("file", "")
            lines.append(f"  - [{fig_file}] {caption}{doi_str}")

    return "\n".join(lines)


def build_db_summary_context() -> str:
    """Load the database summary for recommendation queries."""
    if DB_SUMMARY_PATH.exists():
        return DB_SUMMARY_PATH.read_text(encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_client():
    from openai import OpenAI
    from runtime_config import api_config

    api_key, base_url = api_config()
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)


def query_llm(client, system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Call LLM API and return response text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def query_llm_stream(client, system_prompt: str, user_prompt: str,
                     model: str = DEFAULT_MODEL, chat_history: list = None):
    """Call LLM API with streaming. Yields partial text chunks.

    chat_history: optional list of {"role": "user"/"assistant", "content": str}
                  representing prior turns (injected between system and current user msg).
    """
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        stream=True,
    )
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ---------------------------------------------------------------------------
# Main query pipeline
# ---------------------------------------------------------------------------

def answer_query(query: str, client=None, model: str = DEFAULT_MODEL, verbose: bool = False) -> str:
    """Full RAG pipeline: detect intent → retrieve → generate answer."""

    # 1. Detect intent
    intent = detect_query_intent(query)
    if verbose:
        print(f"  Intent: products={intent['products']}, elements={intent['elements']}, "
              f"structures={intent['structures']}")
        flags = [k for k in ("is_recommendation", "is_comparison", "is_trend", "is_top_n") if intent[k]]
        if flags:
            print(f"  Query type: {', '.join(flags)}")

    # 2. Expand query with English keywords for literature search
    expanded_query = query
    for product in intent["products"]:
        expanded_query += f" {product}"
    for elem in intent["elements"]:
        kws = CATALYST_KEYWORDS.get(elem, [])
        if kws:
            expanded_query += f" {kws[0]} {elem}"
    for struct in intent["structures"]:
        kws = STRUCTURE_KEYWORDS.get(struct, [])
        if kws:
            expanded_query += f" {kws[0]}"

    # 3. Retrieve relevant data
    records = retrieve_records(query, intent, top_k=20)
    if verbose:
        print(f"  Retrieved {len(records)} records")

    lit_entries = retrieve_literature(expanded_query, intent, top_k=5)
    if intent.get("compare_plain_cu"):
        lit_entries = retrieve_comparison_literature(records, per_group=3)
    if intent["is_top_n"]:
        # Numeric leaderboards are authoritative in the structured database.
        # Excluding literature prevents local examples from overriding global ranks.
        lit_entries = []
    if verbose:
        print(f"  Retrieved {len(lit_entries)} literature entries")

    # 3b. Retrieve relevant figures if characterization query
    figures = []
    if intent.get("characterization_types"):
        figures = retrieve_figures(intent, top_k=10)
        if verbose:
            print(f"  Retrieved {len(figures)} figures ({', '.join(intent['characterization_types'])})")

    # 4. Build context
    data_context = build_data_context(records)
    lit_context = build_literature_context(lit_entries)
    ranking_guard = build_ranking_guard(intent)
    figure_context = build_figure_context(figures)

    # For recommendation queries, also include DB summary
    extra_context = ""
    if intent["is_recommendation"]:
        extra_context = "\n### 数据库全局统计\n\n" + build_db_summary_context()

    # 5. Build user prompt
    user_prompt = f"""用户提问：{query}

以下是从知识库中检索到的相关信息：

{data_context}
{ranking_guard}
{lit_context}
{figure_context}
{extra_context}

请基于以上数据和文献信息回答用户的问题。"""

    if verbose:
        print(f"  Prompt length: {len(user_prompt)} chars")

    # 6. Call LLM
    if client is None:
        client = get_client()

    answer = query_llm(client, SYSTEM_PROMPT, user_prompt, model=model)
    return answer


def answer_query_stream(query: str, client=None, model: str = DEFAULT_MODEL,
                        chat_history: list = None):
    """Streaming version of answer_query. Yields (debug_info, partial_answer) tuples.

    chat_history: list of (user_query, assistant_answer) string tuples from prior turns.
    """

    # 1. Detect intent + update user memory
    intent = detect_query_intent(query)
    memory = get_user_memory()
    memory.record_query(query, intent)

    # 2. Expand query
    expanded_query = query
    for product in intent["products"]:
        expanded_query += f" {product}"
    for elem in intent["elements"]:
        kws = CATALYST_KEYWORDS.get(elem, [])
        if kws:
            expanded_query += f" {kws[0]} {elem}"
    for struct in intent["structures"]:
        kws = STRUCTURE_KEYWORDS.get(struct, [])
        if kws:
            expanded_query += f" {kws[0]}"

    # 3. Retrieve
    records = retrieve_records(query, intent, top_k=20)
    lit_entries = retrieve_literature(expanded_query, intent, top_k=5)
    if intent.get("compare_plain_cu"):
        lit_entries = retrieve_comparison_literature(records, per_group=3)
    if intent["is_top_n"]:
        lit_entries = []
    figures = []
    if intent.get("characterization_types"):
        figures = retrieve_figures(intent, top_k=10)

    # 4. Build debug info
    intent_str = []
    if intent["products"]:
        intent_str.append(f"产物: {', '.join(intent['products'])}")
    if intent["elements"]:
        intent_str.append(f"元素: {', '.join(intent['elements'])}")
    if intent["structures"]:
        intent_str.append(f"结构: {', '.join(intent['structures'])}")
    flags = []
    if intent["is_recommendation"]:
        flags.append("推荐")
    if intent["is_comparison"]:
        flags.append("对比")
    if intent["is_trend"]:
        flags.append("趋势")
    if intent["is_top_n"]:
        flags.append("排名")
    if intent.get("is_synthesis"):
        flags.append("合成方法")
    if intent.get("characterization_types"):
        flags.append(f"表征: {', '.join(intent['characterization_types'])}")
    if flags:
        intent_str.append(f"类型: {', '.join(flags)}")

    debug_parts = [
        f"意图识别: {' | '.join(intent_str) if intent_str else '通用查询'}",
        f"检索到: {len(records)} 条数据记录, {len(lit_entries)} 篇文献, {len(figures)} 张相关图片",
    ]
    if not records.empty:
        top3 = records.head(3)
        debug_parts.append("Top-3 记录:")
        for _, r in top3.iterrows():
            debug_parts.append(f"  {r['catalyst_name'][:40]} → {r['product']}: FE={r['fe_pct']:.1f}%")
    debug_info = "\n".join(debug_parts)

    # 5. Build context & prompt
    data_context = build_data_context(records)
    lit_context = build_literature_context(lit_entries)
    ranking_guard = build_ranking_guard(intent)
    figure_context = build_figure_context(figures)
    extra_context = ""
    if intent["is_recommendation"]:
        extra_context = "\n### 数据库全局统计\n\n" + build_db_summary_context()

    # Build user profile context
    profile_summary = memory.get_profile_summary()
    profile_block = ""
    if profile_summary:
        profile_block = f"\n### 用户画像\n{profile_summary}\n"

    user_prompt = f"""用户提问：{query}

以下是从知识库中检索到的相关信息：

{data_context}
{ranking_guard}
{lit_context}
{figure_context}
{extra_context}
{profile_block}
请基于以上数据和文献信息回答用户的问题。如果用户的问题与之前的对话有关联，请结合上下文回答。"""

    # 6. Build prior conversation turns for multi-turn context
    llm_history = []
    if chat_history and not intent["is_top_n"]:
        # Keep last N turns to stay within token limits
        recent = chat_history[-MAX_HISTORY_TURNS:]
        for user_q, assistant_a in recent:
            llm_history.append({"role": "user", "content": user_q})
            llm_history.append({"role": "assistant", "content": assistant_a})

    debug_parts.append(f"对话上下文: {len(llm_history)//2} 轮历史")
    if profile_summary:
        debug_parts.append(f"用户画像: {profile_summary[:80]}...")
    debug_info = "\n".join(debug_parts)

    # 7. Stream LLM response
    if client is None:
        client = get_client()

    partial = ""
    for chunk in query_llm_stream(client, SYSTEM_PROMPT, user_prompt,
                                   model=model, chat_history=llm_history):
        partial += chunk
        yield debug_info, partial


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode(model: str = DEFAULT_MODEL):
    """Run interactive Q&A loop."""
    print("=" * 60)
    print("  CO2RR 催化剂知识库查询系统")
    print("=" * 60)
    print(f"  数据库: {len(load_records())} 条记录 / {len(load_knowledge_base())} 篇文献条目")
    print(f"  模型: {model}")
    print(f"  输入问题开始查询，输入 'quit' 退出")
    print("=" * 60)

    client = get_client()

    # Example queries
    print("\n示例查询:")
    print("  1. 哪些Cu基催化剂生成C2H4的FE最高？")
    print("  2. 推荐一个CO2到HCOOH的高效催化剂，合成要简单")
    print("  3. oxide-derived Cu 和 metallic Cu 在C2产物上有什么差异？")
    print("  4. Cu-Sn 合金催化剂的产物分布是怎样的？")
    print("  5. 单原子Cu催化剂的研究进展？")
    print()

    while True:
        try:
            query = input("🔍 提问: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("退出。")
            break

        print()
        try:
            answer = answer_query(query, client=client, model=model, verbose=True)
            print(f"\n{'─' * 60}")
            print(answer)
            print(f"{'─' * 60}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CO2RR Catalyst Knowledge Base Query")
    parser.add_argument("--query", type=str, default=None, help="Single query (non-interactive)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.query:
        answer = answer_query(args.query, model=args.model, verbose=args.verbose)
        print(answer)
    else:
        interactive_mode(model=args.model)


if __name__ == "__main__":
    main()
