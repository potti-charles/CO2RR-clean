"""Ensemble figure classification: multi-label + multi-model voting.

Strategy:
1. Rule-based classifier (text/caption keywords) → high-precision labels
2. Qwen3-VL-235B (strongest VL) → image+text multi-label
3. Qwen2.5-VL-72B (second VL) → image+text multi-label
4. Merge: union of all labels, ranked by vote count

Key improvement: supports MULTI-LABEL classification for composite figures
(e.g., a figure containing both XRD and XPS subplots).

Usage:
    python reclassify_figures_ensemble.py                    # All figures
    python reclassify_figures_ensemble.py --limit 20         # Test
    python reclassify_figures_ensemble.py --only_other       # Only 'other'
"""

import argparse
import base64
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import List

FIGURE_INDEX = Path("outputs/figure_index.json")
FIGURES_DIR = Path("outputs/figures")

CATEGORIES = [
    "xrd", "electron_microscopy", "spectroscopy", "electrochemistry",
    "faradaic_efficiency", "current_density", "stability", "morphology_scheme",
]

# ---------------------------------------------------------------------------
# Rule-based classifier (high precision)
# ---------------------------------------------------------------------------

RULE_KEYWORDS = {
    "xrd": ["xrd", "x-ray diffraction", "diffraction pattern", "2θ", "2-theta",
            "bragg", "crystalline phase", "jcpds", "icdd"],
    "electron_microscopy": ["sem ", "tem ", "haadf", "hrtem", "stem", "edx", "eds map",
                            "saed", "scanning electron", "transmission electron",
                            "electron microscop", "lattice fringe", "nanoparticle morpholog"],
    "spectroscopy": ["xps", "x-ray photoelectron", "ftir", "raman", "uv-vis", "xanes",
                     "exafs", "infrared", "auger", "nmr", "epr", "binding energy",
                     "wavenumber", "photoluminescence"],
    "faradaic_efficiency": ["faradaic efficiency", "faradaic eff", "fe(%)", "fe (%)",
                            "faradic efficiency", "selectivity", "fe%"],
    "current_density": ["current density", "current dens", "partial current",
                        "j-v", "jco", "jhcooh", "geometric current"],
    "stability": ["stability", "long-term", "durability", "cycling test",
                  "chronoamper", "time-on-stream"],
    "morphology_scheme": ["schematic", "illustration", "scheme", "mechanism",
                          "pathway", "diagram", "synthesis procedure", "formation process"],
    "electrochemistry": ["cyclic voltam", " cv ", "linear sweep", " lsv",
                         "eis", "nyquist", "tafel", "impedance", "polarization",
                         "onset potential"],
}


def classify_rule_based(caption: str, page_text: str = "") -> List[str]:
    """Rule-based multi-label classification from text."""
    text = (caption + " " + page_text).lower()
    labels = []
    for cat, keywords in RULE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            labels.append(cat)
    return labels


# ---------------------------------------------------------------------------
# VL model classifier (multi-label)
# ---------------------------------------------------------------------------

MULTI_LABEL_PROMPT = """You are classifying a scientific figure from a CO2 electroreduction (CO2RR) paper.

IMPORTANT: Many figures in papers contain MULTIPLE sub-panels (e.g., XRD + XPS + TEM in one figure).
You must identify ALL types present in this image.

## Categories
- xrd: X-ray diffraction patterns (2θ vs intensity)
- electron_microscopy: SEM, TEM, HAADF-STEM, EDS mapping, SAED
- spectroscopy: XPS, Raman, FTIR, UV-Vis, XANES, EXAFS spectra
- electrochemistry: CV, LSV, EIS/Nyquist, Tafel, polarization curves
- faradaic_efficiency: FE% bar charts or selectivity plots
- current_density: Current density plots (j vs V or j vs time)
- stability: Chronoamperometry (>1h), long-term durability
- morphology_scheme: Schematic diagrams, reaction mechanisms, DFT structure models
- other: Only if NONE of the above apply (photos, TOC graphics)

## Text context
Caption: {caption}
PDF: {pdf_name}

## Instructions
List ALL categories that apply to any sub-panel in this figure.
Reply with comma-separated category names, nothing else.
Example: xrd, spectroscopy, electron_microscopy"""


def classify_vl(client, model: str, img_path: str, caption: str, pdf_name: str) -> List[str]:
    """VL model multi-label classification."""
    img_b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
    prompt = MULTI_LABEL_PROMPT.format(
        caption=caption[:300] if caption else "(no caption)",
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
                max_tokens=60,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip().lower()
            # Handle thinking model output: extract content after </think>
            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            labels = []
            for cat in CATEGORIES:
                if cat in raw:
                    labels.append(cat)
            return labels if labels else ["other"]
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(3 * (attempt + 1))
            else:
                print(f"    [{model[:20]}] Error: {e}")
                if attempt < 2:
                    time.sleep(2)
    return []


# ---------------------------------------------------------------------------
# Ensemble voting
# ---------------------------------------------------------------------------

def ensemble_classify(rule_labels: List[str], vl1_labels: List[str],
                      vl2_labels: List[str]) -> List[str]:
    """Merge labels from 3 classifiers using voting.

    Logic:
    - A label appearing in >=2 classifiers is accepted (majority vote)
    - Rule-based labels get bonus weight (high precision)
    - If no label reaches 2 votes, take the strongest VL model's labels
    """
    votes = Counter()

    # Rule-based gets weight 1.5 (trusted for keyword matches)
    for lab in rule_labels:
        votes[lab] += 1.5

    # VL models each get weight 1
    for lab in vl1_labels:
        votes[lab] += 1.0
    for lab in vl2_labels:
        votes[lab] += 1.0

    # Accept labels with >= 2 votes (at least 2 sources agree, or rule+1 VL)
    accepted = [cat for cat, v in votes.items() if v >= 2.0 and cat != "other"]

    if not accepted:
        # Fallback: take labels with >= 1.5 votes (rule-based alone, or both VLs agree)
        accepted = [cat for cat, v in votes.items() if v >= 1.5 and cat != "other"]

    if not accepted:
        # Last resort: strongest VL model's labels
        accepted = [lab for lab in vl1_labels if lab != "other"]

    if not accepted:
        accepted = ["other"]

    # Sort by vote count (primary label first)
    accepted.sort(key=lambda x: -votes.get(x, 0))
    return accepted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_client():
    from openai import OpenAI
    from runtime_config import api_config

    api_key, base_url = api_config()
    return OpenAI(api_key=api_key, base_url=base_url, timeout=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", default="Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument("--model2", default="Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only_other", action="store_true",
                        help="Only reclassify current 'other' figures")
    parser.add_argument("--suspicious", action="store_true",
                        help="Only reclassify 'other' + no/short caption figures (high-risk)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    client = get_client()

    # Load index
    index = json.loads(FIGURE_INDEX.read_text(encoding="utf-8"))

    if args.suspicious:
        targets = [i for i, f in enumerate(index)
                   if f.get("category") == "other"
                   or not f.get("caption")
                   or len(f.get("caption", "")) < 20]
    elif args.only_other:
        targets = [i for i, f in enumerate(index) if f.get("category") == "other"]
    else:
        targets = list(range(len(index)))

    print(f"Total figures: {len(index)}, targets: {len(targets)}")
    print(f"Models: {args.model1} + {args.model2} + rule-based")
    print(f"Mode: {'dry run' if args.dry_run else 'live'}")

    if args.limit:
        targets = targets[:args.limit]
        print(f"Processing first {args.limit}")

    changed = 0
    multi_label_count = 0

    for idx_i, fig_idx in enumerate(targets):
        fig = index[fig_idx]
        img_path = fig.get("output_path", f"outputs/figures/{fig['category']}/{fig['file']}")
        caption = fig.get("caption", "") or ""
        pdf_name = fig.get("pdf", "") or ""
        old_cat = fig["category"]

        if not Path(img_path).exists():
            continue

        # 1. Rule-based
        rule_labels = classify_rule_based(caption)

        # 2. VL model 1 (strongest)
        vl1_labels = classify_vl(client, args.model1, img_path, caption, pdf_name)
        time.sleep(0.5)

        # 3. VL model 2
        vl2_labels = classify_vl(client, args.model2, img_path, caption, pdf_name)
        time.sleep(0.5)

        # 4. Ensemble
        final_labels = ensemble_classify(rule_labels, vl1_labels, vl2_labels)
        primary_cat = final_labels[0]

        if len(final_labels) > 1:
            multi_label_count += 1

        if primary_cat != old_cat or len(final_labels) > 1:
            changed += 1

            if not args.dry_run:
                # Move file if primary category changed
                if primary_cat != old_cat:
                    new_path = FIGURES_DIR / primary_cat / fig["file"]
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    if Path(img_path).exists():
                        shutil.move(str(img_path), str(new_path))
                    index[fig_idx]["output_path"] = str(new_path)

                index[fig_idx]["category"] = primary_cat
                index[fig_idx]["categories"] = final_labels  # multi-label

            label_str = ", ".join(final_labels)
            if primary_cat != old_cat:
                print(f"  [{idx_i+1}/{len(targets)}] {fig['file'][:50]} : {old_cat} → {label_str}")
            else:
                print(f"  [{idx_i+1}/{len(targets)}] {fig['file'][:50]} : +labels [{label_str}]")

        # Progress
        if (idx_i + 1) % 50 == 0:
            print(f"\n  === Progress: {idx_i+1}/{len(targets)} (changed={changed}) ===\n")

    # Save
    if not args.dry_run and changed > 0:
        FIGURE_INDEX.write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\n{'='*50}")
    print(f"Done: {changed} changed, {multi_label_count} multi-label figures")

    # Final distribution
    cats = Counter(f["category"] for f in index)
    multi = sum(1 for f in index if len(f.get("categories", [])) > 1)
    print(f"\nFinal distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat:25s}: {count}")
    print(f"\nMulti-label figures: {multi}")


if __name__ == "__main__":
    main()
