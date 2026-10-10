"""Reclassify 'other' category figures using vision + text combined approach.

Uses the strongest VL model (Qwen3-VL-235B) with both image content AND
caption/page text to maximize classification accuracy.

Usage:
    python reclassify_figures_vl.py                    # Process all 'other' figures
    python reclassify_figures_vl.py --limit 10         # Test with 10 figures
    python reclassify_figures_vl.py --model Qwen2.5-VL-72B-Instruct  # Use different model
"""

import argparse
import base64
import json
import os
import shutil
import time
from pathlib import Path

FIGURE_INDEX = Path("outputs/figure_index.json")
FIGURES_DIR = Path("outputs/figures")

CATEGORIES = [
    "xrd", "electron_microscopy", "spectroscopy", "electrochemistry",
    "faradaic_efficiency", "current_density", "stability", "morphology_scheme", "other",
]

CLASSIFY_PROMPT_TEMPLATE = """You are classifying scientific figures from CO2 electroreduction (CO2RR) research papers.

## Task
Look at the image AND read the text context below, then classify this figure into exactly ONE category.

## Categories
- xrd: X-ray diffraction patterns (2θ vs intensity plots, crystal structure peaks)
- electron_microscopy: SEM, TEM, HAADF-STEM, EDS elemental mapping, SAED patterns, microscopy images
- spectroscopy: XPS, Raman, FTIR, UV-Vis, XANES, EXAFS spectra (binding energy / wavenumber plots)
- electrochemistry: CV, LSV, EIS/Nyquist, Tafel plots, polarization curves, i-t curves (not FE%)
- faradaic_efficiency: Faradaic efficiency bar charts or line plots (FE% vs potential/time)
- current_density: Partial current density plots, total current density vs potential
- stability: Chronoamperometry, long-term durability tests, cycling stability (current/FE vs time, >1h)
- morphology_scheme: Schematic diagrams, reaction mechanism illustrations, synthesis flowcharts, crystal structure models
- other: Table of contents graphics, photographs of equipment, unrelated decorative images

## Text context from the paper
Caption: {caption}
PDF filename: {pdf_name}

## Instructions
1. First examine the IMAGE carefully - what type of plot/diagram/image is it?
2. Then check if the text context confirms or clarifies the image type.
3. Reply with ONLY the category name, nothing else."""


def get_client():
    from openai import OpenAI
    from runtime_config import api_config

    api_key, base_url = api_config()
    return OpenAI(api_key=api_key, base_url=base_url, timeout=90)


def classify_image(client, model: str, img_path: str, caption: str, pdf_name: str) -> str:
    """Send image + text context to VL model and return category."""
    img_b64 = base64.b64encode(Path(img_path).read_bytes()).decode()

    prompt = CLASSIFY_PROMPT_TEMPLATE.format(
        caption=caption[:300] if caption else "(no caption available)",
        pdf_name=pdf_name,
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }],
                max_tokens=30,
                temperature=0,
            )
            result = resp.choices[0].message.content.strip().lower()
            # Validate against known categories
            for cat in CATEGORIES:
                if cat in result:
                    return cat
            return "other"
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 3 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(2)
    return "other"


def main():
    parser = argparse.ArgumentParser(description="Reclassify 'other' figures with VL model + text")
    parser.add_argument("--model", default="Qwen3-VL-235B-A22B-Instruct",
                        help="VL model to use (default: strongest available)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Don't move files or update index")
    args = parser.parse_args()

    client = get_client()

    # Load index
    index = json.loads(FIGURE_INDEX.read_text(encoding="utf-8"))
    others = [i for i, f in enumerate(index) if f.get("category") == "other"]
    print(f"Total figures: {len(index)}, 'other' to reclassify: {len(others)}")
    print(f"Model: {args.model}")
    print(f"Mode: {'dry run' if args.dry_run else 'live (will move files)'}")
    print()

    if args.limit:
        others = others[:args.limit]
        print(f"Processing first {args.limit}")

    reclassified = 0
    still_other = 0
    errors = 0
    stats = {cat: 0 for cat in CATEGORIES}

    for idx_i, fig_idx in enumerate(others):
        fig = index[fig_idx]
        img_path = fig.get("output_path", f"outputs/figures/other/{fig['file']}")
        caption = fig.get("caption", "") or ""
        pdf_name = fig.get("pdf", "") or ""

        if not Path(img_path).exists():
            errors += 1
            continue

        # Classify with image + text
        new_cat = classify_image(client, args.model, img_path, caption, pdf_name)
        stats[new_cat] += 1

        if new_cat != "other":
            reclassified += 1

            if not args.dry_run:
                # Move file
                new_path = FIGURES_DIR / new_cat / fig["file"]
                new_path.parent.mkdir(parents=True, exist_ok=True)
                if Path(img_path).exists():
                    shutil.move(str(img_path), str(new_path))

                # Update index
                index[fig_idx]["category"] = new_cat
                index[fig_idx]["output_path"] = str(new_path)

            print(f"  [{idx_i+1}/{len(others)}] {fig['file'][:55]} : other → {new_cat}")
        else:
            still_other += 1
            if (idx_i + 1) % 50 == 0:
                print(f"  [{idx_i+1}/{len(others)}] ... ({still_other} still other)")

        # Rate limiting: ~1 req/s for large model
        time.sleep(0.8)

        # Progress
        if (idx_i + 1) % 100 == 0:
            print(f"\n  === Progress: {idx_i+1}/{len(others)} "
                  f"(reclassified={reclassified}, still_other={still_other}) ===\n")

    # Save updated index
    if not args.dry_run and reclassified > 0:
        FIGURE_INDEX.write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nIndex saved to {FIGURE_INDEX}")

    print(f"\n{'='*50}")
    print(f"Done: {reclassified} reclassified, {still_other} remain 'other', {errors} errors")
    print(f"\nReclassification distribution:")
    for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {cat:25s}: {count}")


if __name__ == "__main__":
    main()
