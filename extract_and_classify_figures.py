"""Extract figures from PDFs, classify by type, and build a figure knowledge base.

Categories:
  - xrd: X-ray diffraction patterns
  - electron_microscopy: SEM, TEM, HAADF, STEM, EDX mapping
  - spectroscopy: XPS, FTIR, Raman, UV-Vis, XANES, EXAFS
  - electrochemistry: CV, LSV, i-t curves, EIS/Nyquist
  - faradaic_efficiency: FE bar/line charts, selectivity plots
  - current_density: partial current density, j-V curves
  - stability: long-term stability tests, chronoamperometry
  - morphology_scheme: schematic illustrations, synthesis diagrams
  - other: unclassified

Usage:
    python extract_and_classify_figures.py                  # Process all PDFs
    python extract_and_classify_figures.py --limit 5        # First 5 PDFs
    python extract_and_classify_figures.py --pdf foo.pdf    # Single PDF
    python extract_and_classify_figures.py --classify_only  # Re-classify existing figures
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_DIR = "gold_corpus/pdfs"
OUTPUT_BASE = "outputs/figures"
INDEX_FILE = "outputs/figure_index.json"
OCR_MD_DIR = "outputs/ocr_markdown"

CATEGORIES = [
    "xrd",
    "electron_microscopy",
    "spectroscopy",
    "electrochemistry",
    "faradaic_efficiency",
    "current_density",
    "stability",
    "morphology_scheme",
    "other",
]

# Minimum image size to keep (skip tiny icons/logos)
MIN_WIDTH = 150
MIN_HEIGHT = 150
MIN_AREA = 40000  # width * height

# Classification prompt
CLASSIFY_SYSTEM = """You are a materials science figure classifier for CO2 electroreduction papers.
Given a figure caption from a scientific paper, classify the figure into ONE of these categories:

- xrd: X-ray diffraction patterns (XRD, PXRD, GIXRD)
- electron_microscopy: SEM, TEM, HAADF-STEM, HRTEM, EDX/EDS mapping, SAED
- spectroscopy: XPS, FTIR, Raman, UV-Vis, XANES, EXAFS, Auger, NMR, EPR
- electrochemistry: cyclic voltammetry (CV), linear sweep voltammetry (LSV), i-t curves, EIS/Nyquist, Tafel plots, chronoamperometry (non-stability)
- faradaic_efficiency: Faradaic efficiency charts, product selectivity plots, FE vs potential
- current_density: partial current density plots, total current density, j-V curves, geometric current density
- stability: long-term stability tests (>1h), chronoamperometry for durability, cycling tests
- morphology_scheme: schematic illustrations, synthesis diagrams, mechanism cartoons, reaction pathways
- other: photos, TOC graphics, or anything not fitting above

Rules:
- If a figure contains MULTIPLE panels with different types, choose the DOMINANT type.
- If a figure shows both FE and current density, choose "faradaic_efficiency".
- If a figure shows both electrochemistry setup and results, choose "electrochemistry".
- Return ONLY the category name, nothing else."""


def get_api_config() -> tuple:
    from runtime_config import api_config
    return api_config()


# ---------------------------------------------------------------------------
# Extract figures from PDF
# ---------------------------------------------------------------------------

def extract_page_text(page) -> str:
    """Extract text from a PDF page using PyMuPDF."""
    try:
        return page.get_text("text")
    except Exception:
        return ""


def find_figure_captions_in_text(text: str) -> List[str]:
    """Extract figure captions from page text using regex."""
    captions = []
    # Match "Fig. 1.", "Figure 1.", "Fig 2:" etc followed by description
    for m in re.finditer(
        r"((?:Fig(?:ure|\.)\s*\d+[\.\):]?\s*).{20,500}?)(?:\n\n|\n[A-Z]|\Z)",
        text, re.IGNORECASE | re.DOTALL
    ):
        captions.append(m.group(1).strip())
    # Also match "Scheme 1" etc
    for m in re.finditer(
        r"((?:Scheme\s*\d+[\.\):]?\s*).{20,300}?)(?:\n\n|\n[A-Z]|\Z)",
        text, re.IGNORECASE | re.DOTALL
    ):
        captions.append(m.group(1).strip())
    return captions


def extract_figures_from_pdf(pdf_path: Path, output_dir: Path) -> List[Dict]:
    """Extract images from a PDF file, filter by size, save to disk.
    Also extracts page text and figure captions for classification."""
    doc = fitz.open(str(pdf_path))
    figures = []
    seen_xrefs = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        # Extract text and captions from this page
        page_text = extract_page_text(page)
        page_captions = find_figure_captions_in_text(page_text)

        page_fig_idx = 0
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                pix = fitz.Pixmap(doc, xref)
                # Convert CMYK to RGB
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                w, h = pix.width, pix.height
                if w < MIN_WIDTH or h < MIN_HEIGHT or w * h < MIN_AREA:
                    continue

                # Save as PNG
                fig_name = f"{pdf_path.stem}_p{page_num+1}_img{img_idx}.png"
                fig_path = output_dir / fig_name
                pix.save(str(fig_path))

                # Assign caption from same page if available
                caption = ""
                if page_fig_idx < len(page_captions):
                    caption = page_captions[page_fig_idx]
                page_fig_idx += 1

                figures.append({
                    "file": fig_name,
                    "path": str(fig_path),
                    "page": page_num + 1,
                    "width": w,
                    "height": h,
                    "pdf": pdf_path.name,
                    "caption": caption,
                    "page_text": page_text[:1000],  # Keep context for classification
                })
            except Exception as e:
                continue

    doc.close()
    return figures


# ---------------------------------------------------------------------------
# Extract captions from OCR markdown
# ---------------------------------------------------------------------------

def extract_captions_from_markdown(md_path: Path) -> List[Dict]:
    """Extract figure captions from OCR markdown."""
    if not md_path.exists():
        return []

    text = md_path.read_text(encoding="utf-8")
    captions = []

    # Pattern 1: ![caption](url) markdown images
    for m in re.finditer(r"!\[([^\]]{20,})\]\([^\)]*\)", text):
        cap = m.group(1).strip()
        if len(cap) > 20:
            captions.append({"caption": cap, "source": "markdown_img"})

    # Pattern 2: "Fig. X" or "Figure X" followed by description
    for m in re.finditer(
        r"(?:Fig(?:ure|\.)\s*\d+[\.\):]?\s*)(.{30,500}?)(?:\n\n|\n---)",
        text, re.IGNORECASE | re.DOTALL
    ):
        cap = m.group(0).strip()
        # Skip if already captured via markdown image
        if not any(cap[:50] in c["caption"][:80] for c in captions):
            captions.append({"caption": cap, "source": "fig_reference"})

    # Pattern 3: "Scheme X" descriptions
    for m in re.finditer(
        r"(?:Scheme\s*\d+[\.\):]?\s*)(.{30,500}?)(?:\n\n|\n---)",
        text, re.IGNORECASE | re.DOTALL
    ):
        cap = m.group(0).strip()
        captions.append({"caption": cap, "source": "scheme_reference"})

    return captions


# ---------------------------------------------------------------------------
# Match figures to captions (heuristic: by page number)
# ---------------------------------------------------------------------------

def match_figures_to_captions(
    figures: List[Dict], captions: List[Dict]
) -> List[Dict]:
    """Best-effort matching of extracted images to captions."""
    # Simple approach: assign captions sequentially to figures
    for i, fig in enumerate(figures):
        if i < len(captions):
            fig["caption"] = captions[i]["caption"]
        else:
            fig["caption"] = ""
    return figures


# ---------------------------------------------------------------------------
# Classify figures using LLM
# ---------------------------------------------------------------------------

def classify_caption(api_key: str, base_url: str, caption: str) -> str:
    """Classify a figure caption into a category using LLM."""
    if not caption or len(caption) < 10:
        return "other"

    # Quick rule-based pre-classification for obvious cases
    cap_lower = caption.lower()

    # Rule-based shortcuts
    rules = [
        (["xrd", "x-ray diffraction", "diffraction pattern", "2θ", "2-theta"], "xrd"),
        (["sem ", "tem ", "haadf", "hrtem", "stem", "edx", "eds map", "saed",
          "scanning electron", "transmission electron", "electron microscop"], "electron_microscopy"),
        (["xps", "x-ray photoelectron", "ftir", "raman", "uv-vis", "xanes", "exafs",
          "infrared", "auger", "nmr", "epr"], "spectroscopy"),
        (["faradaic efficiency", "faradaic eff", " fe ", "fe(%)", "fe(%)"], "faradaic_efficiency"),
        (["current density", "current dens", "partial current", "j-v", "jco", "jhcooh"], "current_density"),
        (["stability", "long-term", "durability", "cycling test", "chronoamper"], "stability"),
        (["schematic", "illustration", "scheme", "mechanism", "pathway", "diagram"], "morphology_scheme"),
        (["cyclic voltam", " cv ", "linear sweep", " lsv", "eis", "nyquist", "tafel",
          "impedance", "polarization"], "electrochemistry"),
    ]

    for keywords, category in rules:
        if any(kw in cap_lower for kw in keywords):
            return category

    # Fall back to LLM for ambiguous cases
    url = f"{base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "DeepSeek-V3.2",
        "messages": [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": f"Caption: {caption[:500]}"},
        ],
        "max_tokens": 20,
        "temperature": 0,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()["choices"][0]["message"]["content"].strip().lower()
            # Validate
            for cat in CATEGORIES:
                if cat in result:
                    return cat
    except Exception:
        pass

    return "other"


# ---------------------------------------------------------------------------
# Classify using rule-based on full text (no caption needed)
# ---------------------------------------------------------------------------

def classify_by_context(pdf_name: str, page_num: int, md_path: Path) -> str:
    """Try to classify based on surrounding text in the OCR markdown."""
    if not md_path.exists():
        return "other"

    text = md_path.read_text(encoding="utf-8")

    # Find text around the page marker
    page_marker = f"<!-- Page {page_num} -->"
    idx = text.find(page_marker)
    if idx < 0:
        return "other"

    # Get ~2000 chars around the page
    context = text[max(0, idx - 500):idx + 1500].lower()

    # Same rule-based approach
    rules = [
        (["xrd", "x-ray diffraction", "diffraction pattern"], "xrd"),
        (["sem", "tem", "haadf", "hrtem", "edx", "eds"], "electron_microscopy"),
        (["xps", "ftir", "raman", "xanes", "exafs"], "spectroscopy"),
        (["faradaic efficiency", "faradaic eff", "fe(%)"], "faradaic_efficiency"),
        (["current density", "partial current"], "current_density"),
        (["stability", "long-term", "durability"], "stability"),
        (["schematic", "illustration"], "morphology_scheme"),
        (["cyclic voltam", "linear sweep", "eis", "nyquist", "tafel"], "electrochemistry"),
    ]

    scores = {cat: 0 for cat in CATEGORIES}
    for keywords, category in rules:
        for kw in keywords:
            scores[category] += context.count(kw)

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return "other"


# ---------------------------------------------------------------------------
# Get DOI for a PDF
# ---------------------------------------------------------------------------

def load_doi_map() -> Dict[str, str]:
    """Load PDF filename → DOI mapping from knowledge_base.json (primary) or CSV (fallback)."""
    doi_map = {}
    # Primary: knowledge_base.json has pdf_file + doi for all 516 PDFs
    kb_path = Path("outputs/knowledge_base.json")
    if kb_path.exists():
        import json as _json
        kb = _json.loads(kb_path.read_text(encoding="utf-8"))
        for entry in kb:
            if entry.get("pdf_file") and entry.get("doi"):
                doi_map[entry["pdf_file"]] = entry["doi"]
    # Fallback: structured_records.csv
    if not doi_map:
        csv_path = Path("outputs/structured_records.csv")
        if csv_path.exists():
            import csv
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("pdf_file") and row.get("doi"):
                        doi_map[row["pdf_file"]] = row["doi"]
    return doi_map


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_all(pdf_dir: str, output_base: str, limit: Optional[int] = None,
                single_pdf: Optional[str] = None, classify_only: bool = False):
    api_key, base_url = get_api_config()
    doi_map = load_doi_map()

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Create category subdirectories
    for cat in CATEGORIES:
        (output_base / cat).mkdir(exist_ok=True)

    # Raw extraction dir
    raw_dir = output_base / "_raw"
    raw_dir.mkdir(exist_ok=True)

    # Load existing index
    index_path = Path(INDEX_FILE)
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = []

    indexed_files = {entry["file"] for entry in index}

    # Collect PDFs
    if single_pdf:
        pdf_files = [Path(single_pdf)]
    else:
        pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
        if limit:
            pdf_files = pdf_files[:limit]

    print(f"Processing {len(pdf_files)} PDFs → {output_base}/")

    total_extracted = 0
    total_classified = 0

    for i, pdf_path in enumerate(pdf_files):
        pdf_name = pdf_path.name
        doi = doi_map.get(pdf_name, "")

        if not classify_only:
            # Extract figures
            figures = extract_figures_from_pdf(pdf_path, raw_dir)
            if not figures:
                continue

            print(f"[{i+1}/{len(pdf_files)}] {pdf_name}: {len(figures)} figures")
            total_extracted += len(figures)

            # Classify and organize
            for fig in figures:
                if fig["file"] in indexed_files:
                    continue

                # Classify: try caption first, then page text, then OCR md
                caption = fig.get("caption", "")
                page_text = fig.get("page_text", "")

                if caption and len(caption) > 20:
                    category = classify_caption(api_key, base_url, caption)
                elif page_text and len(page_text) > 50:
                    # Use page text as pseudo-caption for rule-based classification
                    category = classify_caption(api_key, base_url, page_text[:500])
                else:
                    md_path = Path(OCR_MD_DIR) / f"{pdf_path.stem}.md"
                    category = classify_by_context(
                        pdf_name, fig["page"], md_path
                    )

                # Copy to category folder
                src = Path(fig["path"])
                dst = output_base / category / fig["file"]
                if src.exists():
                    import shutil
                    shutil.copy2(str(src), str(dst))

                # Add to index
                entry = {
                    "file": fig["file"],
                    "category": category,
                    "caption": caption[:500] if caption else "",
                    "pdf": pdf_name,
                    "doi": doi,
                    "page": fig["page"],
                    "width": fig["width"],
                    "height": fig["height"],
                }
                index.append(entry)
                indexed_files.add(fig["file"])
                total_classified += 1

    # Save index
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print summary
    from collections import Counter
    cat_counts = Counter(e["category"] for e in index)
    print(f"\n{'='*50}")
    print(f"Total: {total_extracted} extracted, {total_classified} classified")
    print(f"Index: {len(index)} figures total")
    print(f"\nCategory distribution:")
    for cat in CATEGORIES:
        print(f"  {cat:25s}: {cat_counts.get(cat, 0):4d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default=PDF_DIR)
    parser.add_argument("--output", default=OUTPUT_BASE)
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--classify_only", action="store_true")
    args = parser.parse_args()

    process_all(
        pdf_dir=args.pdf_dir,
        output_base=args.output,
        limit=args.limit,
        single_pdf=args.pdf,
        classify_only=args.classify_only,
    )
