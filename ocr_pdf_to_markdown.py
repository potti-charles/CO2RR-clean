"""Batch OCR: Convert PDFs to Markdown using DeepSeek-OCR API.

Usage:
    python ocr_pdf_to_markdown.py                          # Process all PDFs
    python ocr_pdf_to_markdown.py --limit 5                # Process first 5
    python ocr_pdf_to_markdown.py --pdf gold_corpus/pdfs/61_807.pdf  # Single file
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import fitz
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PDF_DIR = "gold_corpus/pdfs"
DEFAULT_OUTPUT_DIR = "outputs/ocr_markdown"
OCR_MODEL = "DeepSeek-OCR"
DPI = 100  # DeepSeek-OCR has strict token limit; 100 DPI works reliably
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def get_api_config() -> tuple:
    from runtime_config import api_config
    return api_config()


# ---------------------------------------------------------------------------
# PDF to images
# ---------------------------------------------------------------------------

def pdf_page_to_base64(pdf_path: str, page_num: int, dpi: int = DPI) -> str:
    """Convert a single PDF page to base64-encoded PNG."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode()


def get_pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


# ---------------------------------------------------------------------------
# OCR API call
# ---------------------------------------------------------------------------

def ocr_image(api_key: str, base_url: str, img_b64: str,
              max_tokens: int = 4096) -> Optional[str]:
    """Send image to DeepSeek-OCR and return markdown text."""
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Convert this page to markdown."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                # Rate limited
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error {resp.status_code}: {resp.text[:200]}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        except requests.exceptions.Timeout:
            print(f"    Timeout on attempt {attempt + 1}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"    Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return None


# ---------------------------------------------------------------------------
# Process single PDF
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path, output_dir: Path,
                api_key: str, base_url: str,
                dpi: int = DPI) -> Optional[Path]:
    """Convert a single PDF to Markdown via OCR."""
    md_path = output_dir / f"{pdf_path.stem}.md"

    # Skip if already processed
    if md_path.exists() and md_path.stat().st_size > 100:
        return md_path

    n_pages = get_pdf_page_count(str(pdf_path))
    pages_md: List[str] = []

    for page_num in range(n_pages):
        try:
            img_b64 = pdf_page_to_base64(str(pdf_path), page_num, dpi=dpi)
        except Exception as e:
            print(f"    Page {page_num}: image conversion failed: {e}")
            continue

        result = ocr_image(api_key, base_url, img_b64)
        if result:
            pages_md.append(f"<!-- Page {page_num + 1} -->\n{result}")
        else:
            print(f"    Page {page_num}: OCR failed")

        # Small delay to avoid rate limiting
        if page_num < n_pages - 1:
            time.sleep(0.5)

    if not pages_md:
        return None

    # Write markdown
    full_md = f"# {pdf_path.stem}\n\n" + "\n\n---\n\n".join(pages_md)
    md_path.write_text(full_md, encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch OCR PDFs to Markdown")
    parser.add_argument("--pdf_dir", default=DEFAULT_PDF_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf", default=None, help="Process single PDF file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    api_key, base_url = get_api_config()
    if not api_key:
        print("ERROR: No API key found")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDFs
    if args.pdf:
        pdf_files = [Path(args.pdf)]
    else:
        pdf_files = sorted(Path(args.pdf_dir).glob("*.pdf"))
        if args.limit:
            pdf_files = pdf_files[:args.limit]

    print(f"Processing {len(pdf_files)} PDF(s) → {output_dir}/")
    print(f"OCR model: {OCR_MODEL}, DPI: {args.dpi}")

    success = 0
    skipped = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files):
        md_path = output_dir / f"{pdf_path.stem}.md"
        if args.skip_existing and md_path.exists() and md_path.stat().st_size > 100:
            skipped += 1
            continue

        n_pages = get_pdf_page_count(str(pdf_path))
        print(f"[{i+1}/{len(pdf_files)}] {pdf_path.name} ({n_pages} pages)")

        result = process_pdf(pdf_path, output_dir, api_key, base_url, dpi=args.dpi)
        if result:
            print(f"  → {result.name} ({result.stat().st_size / 1024:.0f} KB)")
            success += 1
        else:
            print(f"  → FAILED")
            failed += 1

    print(f"\nDone: {success} success, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
