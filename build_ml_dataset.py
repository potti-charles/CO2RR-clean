"""Build ML-ready feature matrix from catalyst leaderboard CSV."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Element parsing
# ---------------------------------------------------------------------------

# Matches element symbols followed by optional stoichiometry, e.g. Cu2, O, Ag
_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

# Common separators in material names (e.g. Ag-Cu, Cu/ZnO, CuO@C)
_SEP_RE = re.compile(r"[-/@|,]")

VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
}


def parse_elements(material: str) -> dict:
    """Parse element symbols and fractional composition from a material string.

    Examples:
        Cu2O   → {Cu: 0.667, O: 0.333}
        Ag-Cu  → {Ag: 0.5, Cu: 0.5}
    """
    if not material or not isinstance(material, str):
        return {}

    # Split on common separators first, parse each fragment
    fragments = _SEP_RE.split(material)
    raw_counts: dict = defaultdict(float)

    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue
        for m in _ELEM_RE.finditer(frag):
            elem = m.group(1)
            if elem not in VALID_ELEMENTS:
                continue
            count = float(m.group(2)) if m.group(2) else 1.0
            raw_counts[elem] += count

    total = sum(raw_counts.values())
    if total == 0:
        return {}
    return {elem: count / total for elem, count in raw_counts.items()}


# ---------------------------------------------------------------------------
# Product normalization
# ---------------------------------------------------------------------------

PRODUCT_MAP = {
    "hcooh": "formate", "formic acid": "formate", "formate": "formate", "hcoo-": "formate",
    "co": "CO", "carbon monoxide": "CO",
    "c2h4": "ethylene", "ethylene": "ethylene",
    "c2h5oh": "ethanol", "ethanol": "ethanol",
    "ch3oh": "methanol", "methanol": "methanol",
    "ch4": "methane", "methane": "methane",
    "h2": "hydrogen", "hydrogen": "hydrogen",
    "c2h6": "ethane", "ethane": "ethane",
    "c3h7oh": "propanol", "propanol": "propanol", "n-propanol": "propanol",
    "ch3cooh": "acetate", "acetic acid": "acetate", "acetate": "acetate",
    "c2h4o": "acetaldehyde", "acetaldehyde": "acetaldehyde",
    "syngas": "syngas",
}


def normalize_product(product: str) -> str:
    """Map a product string to a canonical name."""
    if not product or not isinstance(product, str):
        return ""
    key = product.strip().lower()
    return PRODUCT_MAP.get(key, product.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML feature matrix from leaderboard CSV.")
    parser.add_argument("--input_csv", default="outputs/catalyst_leaderboard.csv")
    parser.add_argument("--output_csv", default="outputs/ml_dataset.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows from {args.input_csv}")

    # Drop rows without a numeric efficiency
    df = df.dropna(subset=["efficiency_pct"]).copy()
    df = df[df["efficiency_pct"] > 0].copy()
    print(f"After filtering: {len(df)} rows with valid efficiency_pct")

    if df.empty:
        print("No valid data to build features. Exiting.")
        return

    # --- Element composition features ---
    elem_dicts = df["material"].apply(parse_elements)
    all_elements = sorted({e for d in elem_dicts for e in d})
    for elem in all_elements:
        df[f"elem_{elem}"] = elem_dicts.apply(lambda d, e=elem: d.get(e, 0.0))

    # --- Product one-hot features ---
    df["product_norm"] = df["product"].apply(normalize_product)
    all_products = sorted(df["product_norm"].unique())
    for prod in all_products:
        if prod:
            df[f"prod_{prod}"] = (df["product_norm"] == prod).astype(int)

    # --- potential_V as-is (already numeric) ---
    df["potential_V"] = pd.to_numeric(df["potential_V"], errors="coerce")

    # Save
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved ML dataset ({len(df)} rows, {len(df.columns)} cols) → {output_path}")

    # Summary
    elem_cols = [c for c in df.columns if c.startswith("elem_")]
    prod_cols = [c for c in df.columns if c.startswith("prod_")]
    print(f"  Element features ({len(elem_cols)}): {elem_cols}")
    print(f"  Product features ({len(prod_cols)}): {prod_cols}")


if __name__ == "__main__":
    main()
