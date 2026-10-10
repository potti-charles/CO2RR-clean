"""Extract and normalize synthesis methods from gold_corpus, link to structured_records.

Usage:
    python extract_synthesis_methods.py
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical synthesis method categories
# ---------------------------------------------------------------------------

# Map raw keywords → canonical category
SYNTHESIS_CATEGORIES = {
    # Electrochemical
    "electrodeposition": "electrodeposition",
    "electrochemical deposition": "electrodeposition",
    "electroplating": "electrodeposition",
    "galvanostatic deposition": "electrodeposition",
    "potentiostatic deposition": "electrodeposition",
    "electrochemical method": "electrodeposition",
    "anodization": "anodization",
    "anodized": "anodization",
    "electrochemical reduction": "electrochemical reduction",
    "electrochemical oxidation": "electrochemical oxidation",
    "dealloying": "dealloying",
    "de-alloying": "dealloying",
    # Wet chemistry
    "wet chemical method": "wet chemical synthesis",
    "wet chemical synthesis": "wet chemical synthesis",
    "co-precipitation": "co-precipitation",
    "coprecipitation": "co-precipitation",
    "precipitation": "co-precipitation",
    "impregnation": "impregnation",
    "chemical reduction": "chemical reduction",
    "nabh4": "chemical reduction",
    # Thermal
    "hydrothermal": "hydrothermal",
    "solvothermal": "solvothermal",
    "thermal treatment": "thermal treatment",
    "thermal annealing": "thermal annealing",
    "annealing": "thermal annealing",
    "calcination": "calcination",
    "pyrolysis": "pyrolysis",
    "carbonization": "pyrolysis",
    # Deposition
    "sputtering": "sputtering",
    "sputter": "sputtering",
    "pvd": "physical vapor deposition",
    "physical vapor deposition": "physical vapor deposition",
    "physical deposition": "physical vapor deposition",
    "cvd": "chemical vapor deposition",
    "chemical vapor deposition": "chemical vapor deposition",
    "atomic layer deposition": "atomic layer deposition",
    "ald": "atomic layer deposition",
    "evaporation": "thermal evaporation",
    "e-beam": "e-beam evaporation",
    # Sol-gel & template
    "sol-gel": "sol-gel",
    "sol gel": "sol-gel",
    "template": "template method",
    # Mechanical / mixing
    "mechanical mixing": "mechanical mixing",
    "ball milling": "ball milling",
    "grinding": "ball milling",
    # Laser / plasma
    "laser": "laser processing",
    "plasma": "plasma treatment",
    # Other
    "spin coating": "spin coating",
    "drop casting": "drop casting",
    "spray": "spray coating",
    "dip coating": "dip coating",
    "mof": "MOF-derived",
    "metal-organic framework": "MOF-derived",
    "metal organic framework": "MOF-derived",
    "structure control": "structure control",
}


def classify_synthesis(entity_text: str) -> str:
    """Map a raw synthesis entity string to a canonical category."""
    text = entity_text.lower().strip()

    # Try direct match first
    if text in SYNTHESIS_CATEGORIES:
        return SYNTHESIS_CATEGORIES[text]

    # Try keyword matching
    for keyword, category in SYNTHESIS_CATEGORIES.items():
        if keyword in text:
            return category

    # If text is short (< 60 chars), it's likely a clean label — keep as-is
    if len(text) < 60:
        return text.strip()

    # Long text — try to extract method keywords from it
    for keyword, category in SYNTHESIS_CATEGORIES.items():
        if keyword in text:
            return category

    return "other"


def extract_synthesis_from_gold_corpus() -> pd.DataFrame:
    """Extract synthesis methods from gold_corpus.xls and normalize."""
    gold = pd.read_excel("gold_corpus/gold_corpus.xls")
    syn = gold[gold["entity_label"] == "Synthesis method"].copy()

    # Classify each entry
    syn["synthesis_category"] = syn["entity"].apply(classify_synthesis)

    # For each DOI, collect unique synthesis categories
    doi_methods = (
        syn.groupby("doi")["synthesis_category"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )
    doi_methods.columns = ["doi", "synthesis_methods"]

    # Also keep the raw entities for detail
    doi_raw = (
        syn.groupby("doi")["entity"]
        .apply(lambda x: list(set(x.str.strip())))
        .reset_index()
    )
    doi_raw.columns = ["doi", "synthesis_raw"]

    result = doi_methods.merge(doi_raw, on="doi", how="left")

    # Also extract from context field for DOIs without explicit synthesis labels
    all_dois = set(gold["doi"].unique())
    labeled_dois = set(syn["doi"].unique())
    unlabeled_dois = all_dois - labeled_dois

    # For unlabeled DOIs, scan context for synthesis keywords
    extra_rows = []
    unlabeled = gold[gold["doi"].isin(unlabeled_dois)]
    for doi, group in unlabeled.groupby("doi"):
        all_context = " ".join(group["context"].dropna().astype(str)).lower()
        found_methods = []
        for keyword, category in SYNTHESIS_CATEGORIES.items():
            if keyword in all_context:
                found_methods.append(category)
        if found_methods:
            found_methods = sorted(set(found_methods))
            extra_rows.append({
                "doi": doi,
                "synthesis_methods": found_methods,
                "synthesis_raw": [f"[inferred from context: {', '.join(found_methods)}]"],
            })

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        result = pd.concat([result, extra_df], ignore_index=True)

    return result


def main():
    print("Extracting synthesis methods from gold_corpus...")
    syn_df = extract_synthesis_from_gold_corpus()
    print(f"  Found synthesis info for {len(syn_df)} DOIs")

    # Load structured records
    records = pd.read_csv("outputs/structured_records.csv")
    print(f"  Structured records: {len(records)} rows, {records['doi'].nunique()} DOIs")

    # Merge
    # Convert synthesis_methods list to semicolon-separated string for CSV
    syn_df["synthesis_methods_str"] = syn_df["synthesis_methods"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )

    merged = records.merge(
        syn_df[["doi", "synthesis_methods_str"]],
        on="doi",
        how="left",
    )
    merged.rename(columns={"synthesis_methods_str": "synthesis_method"}, inplace=True)

    # Fill NaN with empty string
    merged["synthesis_method"] = merged["synthesis_method"].fillna("")

    # Save updated records
    output_path = Path("outputs/structured_records.csv")
    merged.to_csv(output_path, index=False)

    # Stats
    has_syn = merged[merged["synthesis_method"] != ""]
    print(f"\n  Updated structured_records.csv:")
    print(f"    Total records: {len(merged)}")
    print(f"    Records with synthesis method: {len(has_syn)} ({100*len(has_syn)/len(merged):.1f}%)")
    print(f"    DOIs with synthesis method: {has_syn['doi'].nunique()}")

    # Category distribution
    all_methods = []
    for methods_str in has_syn["synthesis_method"]:
        all_methods.extend(methods_str.split("; "))
    method_counts = Counter(all_methods)
    print(f"\n  Top synthesis method categories:")
    for method, count in method_counts.most_common(15):
        print(f"    {method}: {count}")

    # Also save synthesis-specific file for reference
    syn_df.to_json("outputs/synthesis_methods.json", orient="records",
                   force_ascii=False, indent=2)
    print(f"\n  Saved synthesis_methods.json ({len(syn_df)} entries)")


if __name__ == "__main__":
    main()
