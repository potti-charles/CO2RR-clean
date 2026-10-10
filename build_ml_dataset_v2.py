"""Build ML-ready feature matrix from gold_corpus.xls with Magpie-style features.

Pipeline:
  1. Parse gold_corpus.xls → per-paper structured records (catalyst, product, FE%, conditions)
  2. Parse element composition from catalyst names
  3. Compute Magpie-style features (composition-weighted stats of element properties)
  4. Encode catalyst type, product, electrolyte as categorical features
  5. Output: outputs/ml_dataset_v2.csv ready for XGBoost/RF training

Usage:
  python build_ml_dataset_v2.py
  python build_ml_dataset_v2.py --gold_corpus gold_corpus/gold_corpus.xls --output outputs/ml_dataset_v2.csv
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from element_properties import ELEMENT_DATA, PROPERTY_NAMES

# ---------------------------------------------------------------------------
# Catalyst type labels in gold_corpus (entity_label → category)
# ---------------------------------------------------------------------------

CATALYST_TYPE_MAP = {
    "Cu": "Cu",
    "CuOx": "CuOx",
    "Cu-M": "Cu-M",
    "Cu/C": "Cu/C",
    "composite": "Composite",
    "alloy": "Alloy",
    "Cu-MOF": "Cu-MOF",
    "Cu-MOx": "Cu-MOx",
    "CuMOx": "Cu-MOx",
    "Cu(Ox)-MOx": "Cu-MOx",
    "Cu(Ox)-M(OH)x": "Cu-MOx",
    "CuSx": "CuSx",
    "CuSex": "CuSx",  # group chalcogenides
    "CuPx": "CuSx",   # group pnictides with chalcogenides
    "M+CuOx": "Cu-MOx",
    "Cu molecular complex": "Cu-Complex",
    "Cu-molecular complex": "Cu-Complex",
    "CuNx": "CuNx",
    "CuMSx": "CuSx",
    "Cu-MXene": "Cu-MXene",
    "CuI": "CuOx",
    "atomic level dispersion": "SAC",
}

# Labels that represent catalyst materials (vs conditions/methods)
CATALYST_LABELS = set(CATALYST_TYPE_MAP.keys())

# ---------------------------------------------------------------------------
# Element parsing from material names
# ---------------------------------------------------------------------------

_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
_SEP_RE = re.compile(r"[-/@|,\s]+")

# Words to strip from material names before element parsing
_NOISE_WORDS = {
    "nanoparticles", "nanoparticle", "nanowires", "nanowire", "nanotubes",
    "nanotube", "nanosheets", "nanosheet", "nanofibers", "nanofiber",
    "nanocubes", "nanocube", "nanocrystals", "nanocrystal", "nanorods",
    "nanorod", "nanoflowers", "nanoflower", "nanofoam", "foam", "mesh",
    "dendrites", "dendritic", "porous", "hollow", "aerogel", "catalyst",
    "catalysts", "electrode", "electrodes", "electrocatalyst", "decorated",
    "modified", "derived", "based", "supported", "doped", "enriched",
    "interconnected", "hierarchical", "defective", "amorphous",
    "crystalline", "single", "atom", "site", "sites", "surface",
    "interface", "layer", "film", "thin", "thick", "bulk", "foil",
    "vertical", "graphene", "graphite", "carbon", "fiber", "fibre",
    "3d", "2d", "1d",
}


def clean_material_name(name: str) -> str:
    """Remove noise words and keep only chemical formula parts."""
    if not name or not isinstance(name, str):
        return ""
    # Remove parenthetical descriptions
    name = re.sub(r"\([^)]*\)", "", name)
    # Split and filter
    tokens = _SEP_RE.split(name)
    cleaned = []
    for t in tokens:
        if t.lower() in _NOISE_WORDS:
            continue
        if len(t) > 0:
            cleaned.append(t)
    return " ".join(cleaned)


def parse_elements(material: str) -> Dict[str, float]:
    """Parse element composition fractions from a material string."""
    if not material or not isinstance(material, str):
        return {}
    cleaned = clean_material_name(material)
    raw_counts: Dict[str, float] = defaultdict(float)
    for token in _SEP_RE.split(cleaned):
        for m in _ELEM_RE.finditer(token):
            elem = m.group(1)
            if elem not in ELEMENT_DATA:
                continue
            count = float(m.group(2)) if m.group(2) else 1.0
            raw_counts[elem] += count
    total = sum(raw_counts.values())
    if total == 0:
        return {}
    return {elem: count / total for elem, count in raw_counts.items()}


# ---------------------------------------------------------------------------
# Magpie feature computation
# ---------------------------------------------------------------------------

# Properties to use for Magpie stats (indices into PROPERTY_NAMES)
MAGPIE_PROPS = {
    "atomic_mass": 1,
    "electronegativity": 2,
    "atomic_radius": 3,
    "ionization_energy": 4,
    "electron_affinity": 5,
    "melting_point": 6,
    "valence_electrons": 8,
}


def compute_magpie_features(composition: Dict[str, float]) -> Dict[str, float]:
    """Compute Magpie-style statistical features from element composition.

    For each property, compute: weighted_mean, weighted_std, min, max, range
    Also: n_elements, max_fraction, min_fraction
    """
    features: Dict[str, float] = {}

    if not composition:
        # Return zeros for all features
        for prop_name in MAGPIE_PROPS:
            for stat in ("mean", "std", "min", "max", "range"):
                features[f"magpie_{prop_name}_{stat}"] = 0.0
        features["n_elements"] = 0
        features["max_fraction"] = 0.0
        features["min_fraction"] = 0.0
        return features

    elements = list(composition.keys())
    fractions = np.array([composition[e] for e in elements])

    features["n_elements"] = len(elements)
    features["max_fraction"] = float(fractions.max())
    features["min_fraction"] = float(fractions.min())

    for prop_name, prop_idx in MAGPIE_PROPS.items():
        values = np.array([ELEMENT_DATA[e][prop_idx] for e in elements])
        w_mean = float(np.average(values, weights=fractions))
        w_var = float(np.average((values - w_mean) ** 2, weights=fractions))
        w_std = float(np.sqrt(w_var))
        features[f"magpie_{prop_name}_mean"] = w_mean
        features[f"magpie_{prop_name}_std"] = w_std
        features[f"magpie_{prop_name}_min"] = float(values.min())
        features[f"magpie_{prop_name}_max"] = float(values.max())
        features[f"magpie_{prop_name}_range"] = float(values.max() - values.min())

    return features


# ---------------------------------------------------------------------------
# Product normalization
# ---------------------------------------------------------------------------

PRODUCT_NORM = {
    "hcooh": "HCOOH", "formic acid": "HCOOH", "formate": "HCOOH",
    "hcoo-": "HCOOH", "hcoo": "HCOOH",
    "co": "CO", "carbon monoxide": "CO",
    "c2h4": "C2H4", "ethylene": "C2H4",
    "c2h5oh": "C2H5OH", "ethanol": "C2H5OH",
    "ch3oh": "CH3OH", "methanol": "CH3OH",
    "ch4": "CH4", "methane": "CH4",
    "h2": "H2", "hydrogen": "H2",
    "c2h6": "C2H6", "ethane": "C2H6",
    "c2h6o": "C2H5OH",  # alternate formula for ethanol
    "c3h7oh": "C3H7OH", "propanol": "C3H7OH", "n-propanol": "C3H7OH",
    "1-propanol": "C3H7OH",
    "ch3cooh": "CH3COOH", "acetic acid": "CH3COOH", "acetate": "CH3COOH",
    "ch3cho": "CH3CHO", "acetaldehyde": "CH3CHO",
    "syngas": "syngas",
    "c2+": "C2+",
    "ch30h": "CH3OH", "ch3 oh": "CH3OH",
    "c2h50h": "C2H5OH", "ch3ch2oh": "C2H5OH",
    "c0": "CO",
    "propanel": "C3H7OH", "1-propanol": "C3H7OH",
    "c3h6o": "acetone", "acetone": "acetone",
    "ch3ch2ch2oh": "C3H7OH",
}


def normalize_product(p: str) -> str:
    if not p or not isinstance(p, str):
        return ""
    key = p.strip().lower()
    return PRODUCT_NORM.get(key, p.strip())


# ---------------------------------------------------------------------------
# Voltage parsing
# ---------------------------------------------------------------------------

_VOLTAGE_RE = re.compile(r"(-?\d+\.?\d*)\s*V")


def parse_voltage(v: str) -> Optional[float]:
    """Extract voltage value from a string like '-1.2 V vs. RHE'."""
    if not v or not isinstance(v, str):
        return None
    m = _VOLTAGE_RE.search(v)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Electrolyte normalization
# ---------------------------------------------------------------------------

def normalize_electrolyte(e: str) -> str:
    """Normalize electrolyte string to canonical form."""
    if not e or not isinstance(e, str):
        return "unknown"
    e = e.strip().lower()
    if "khco3" in e or "khco" in e:
        return "KHCO3"
    if "nahco3" in e:
        return "NaHCO3"
    if "koh" in e:
        return "KOH"
    if "naoh" in e:
        return "NaOH"
    if "kcl" in e:
        return "KCl"
    if "k2so4" in e:
        return "K2SO4"
    if "cshco3" in e:
        return "CsHCO3"
    if "h2so4" in e:
        return "H2SO4"
    if "bmim" in e or "ionic" in e or "emim" in e:
        return "ionic_liquid"
    return "other"


# ---------------------------------------------------------------------------
# Gold corpus → structured records
# ---------------------------------------------------------------------------

def extract_records_from_gold(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert entity-level gold_corpus into (catalyst, product, FE%, conditions) records.

    Each paper can produce 1-3 records (one per product slot).
    """
    records = []
    for doi, group in df.groupby("doi"):
        label_map: Dict[str, List[str]] = defaultdict(list)
        for _, row in group.iterrows():
            lbl = str(row["entity_label"]).strip()
            ent = str(row["entity"]).strip() if pd.notna(row["entity"]) else ""
            label_map[lbl].append(ent)

        # Extract catalyst: find the entity_label that's a catalyst type
        catalyst_name = ""
        catalyst_type = ""
        for lbl in label_map:
            if lbl in CATALYST_LABELS:
                # Pick the first (usually the main catalyst name)
                for ent in label_map[lbl]:
                    if len(ent) > 3 and len(ent) < 200:  # skip noise
                        catalyst_name = ent
                        catalyst_type = CATALYST_TYPE_MAP.get(lbl, "Other")
                        break
                if catalyst_name:
                    break

        if not catalyst_name:
            continue

        # Extract conditions
        voltage = None
        for v in label_map.get("Faradaic efficiency Voltage", []):
            voltage = parse_voltage(v)
            if voltage is not None:
                break

        electrolyte = "unknown"
        for e in label_map.get("electrolyte", []):
            electrolyte = normalize_electrolyte(e)
            if electrolyte != "unknown":
                break

        cell_setup = ""
        for c in label_map.get("cell setup", []):
            cell_setup = c.strip().lower()
            break

        current_density = None
        for cd in label_map.get(" Current density", []):
            m = re.search(r"([\d.]+)", str(cd))
            if m:
                current_density = float(m.group(1))
                break

        title = group.iloc[0]["title"] if "title" in group.columns else ""
        year = group.iloc[0]["public_year"] if "public_year" in group.columns else None

        # Extract product-FE pairs (1st, 2nd, 3rd)
        for ordinal in ("1st", "2nd", "3rd"):
            prod_key = f"{ordinal} product"
            fe_key = f"{ordinal} product Faradaic efficiency"

            products = label_map.get(prod_key, [])
            fes = label_map.get(fe_key, [])

            if not products or not fes:
                continue

            product_raw = products[0]
            fe_raw = fes[0]

            product = normalize_product(product_raw)
            if not product:
                continue

            # Parse FE value (stored as fraction 0-1 in gold_corpus)
            try:
                fe_val = float(fe_raw)
            except (ValueError, TypeError):
                continue

            # Validate: gold_corpus stores FE as fraction (0-1)
            if fe_val < 0:
                continue
            if fe_val <= 1.0:
                fe_pct = fe_val * 100.0  # convert to percentage
            elif fe_val <= 100.0:
                fe_pct = fe_val  # already percentage
            else:
                continue  # invalid

            if fe_pct <= 0 or fe_pct > 100:
                continue

            records.append({
                "doi": doi,
                "title": title,
                "year": year,
                "catalyst_name": catalyst_name,
                "catalyst_type": catalyst_type,
                "product_raw": product_raw,
                "product": product,
                "fe_pct": round(fe_pct, 2),
                "voltage_V": voltage,
                "electrolyte": electrolyte,
                "cell_setup": cell_setup,
                "current_density_mA_cm2": current_density,
            })

    return records


def apply_record_corrections(
    records_df: pd.DataFrame, corrections_path: str
) -> pd.DataFrame:
    """Apply literature-verified FE/condition corrections by DOI and product.

    The gold spreadsheet occasionally pairs a product with an FE measured at a
    different potential.  Keeping these corrections in a small, reviewable CSV
    makes database rebuilds reproducible without silently changing the source
    annotation workbook.
    """
    path = Path(corrections_path)
    if not path.exists():
        return records_df

    corrections = pd.read_csv(path)
    required = {"doi", "product", "fe_pct", "voltage_V"}
    missing = required.difference(corrections.columns)
    if missing:
        raise ValueError(
            f"Correction file {path} is missing columns: {sorted(missing)}"
        )

    corrected = records_df.copy()
    for row in corrections.itertuples(index=False):
        mask = (
            corrected["doi"].astype(str).str.lower().eq(str(row.doi).lower())
            & corrected["product"].astype(str).str.lower().eq(
                str(row.product).lower()
            )
        )
        match_count = int(mask.sum())
        if match_count != 1:
            raise ValueError(
                f"Expected one record for DOI={row.doi}, product={row.product}; "
                f"found {match_count}"
            )
        corrected.loc[mask, "fe_pct"] = float(row.fe_pct)
        corrected.loc[mask, "voltage_V"] = float(row.voltage_V)

    return corrected


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(records_df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to the records dataframe."""
    df = records_df.copy()

    # --- Element composition ---
    compositions = df["catalyst_name"].apply(parse_elements)

    # --- Magpie features ---
    magpie_rows = compositions.apply(compute_magpie_features)
    magpie_df = pd.DataFrame(magpie_rows.tolist(), index=df.index)
    df = pd.concat([df, magpie_df], axis=1)

    # --- Element fraction features (top elements in CO2RR) ---
    key_elements = ["Cu", "Ag", "Au", "Zn", "Sn", "In", "Bi", "Pb",
                    "Fe", "Co", "Ni", "Pd", "Pt", "Mo", "C", "N", "O", "S"]
    for elem in key_elements:
        df[f"elem_{elem}"] = compositions.apply(lambda d, e=elem: d.get(e, 0.0))

    # --- Catalyst type one-hot ---
    cat_types = sorted(df["catalyst_type"].unique())
    for ct in cat_types:
        df[f"cattype_{ct}"] = (df["catalyst_type"] == ct).astype(int)

    # --- Product one-hot ---
    products = sorted(df["product"].unique())
    for p in products:
        if p:
            df[f"prod_{p}"] = (df["product"] == p).astype(int)

    # --- Electrolyte one-hot ---
    electrolytes = sorted(df["electrolyte"].unique())
    for e in electrolytes:
        df[f"elyte_{e}"] = (df["electrolyte"] == e).astype(int)

    # --- Voltage (numeric, fill missing with median) ---
    df["voltage_V"] = pd.to_numeric(df["voltage_V"], errors="coerce")

    # --- Cell setup encoding ---
    df["cell_flow"] = df["cell_setup"].apply(
        lambda x: 1 if "flow" in str(x) else 0
    )
    df["cell_htype"] = df["cell_setup"].apply(
        lambda x: 1 if "h-type" in str(x) or "h type" in str(x) else 0
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of feature column names for ML model."""
    prefixes = ("magpie_", "elem_", "cattype_", "prod_", "elyte_", "cell_flow", "cell_htype")
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    if "voltage_V" in df.columns:
        cols.append("voltage_V")
    if "n_elements" in df.columns:
        cols.append("n_elements")
    if "max_fraction" in df.columns:
        cols.append("max_fraction")
    if "min_fraction" in df.columns:
        cols.append("min_fraction")
    return sorted(cols)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML dataset from gold_corpus with Magpie features.")
    parser.add_argument("--gold_corpus", default="gold_corpus/gold_corpus.xls")
    parser.add_argument("--output", default="outputs/ml_dataset_v2.csv")
    parser.add_argument("--records_output", default="outputs/structured_records.csv",
                        help="Save intermediate structured records before feature engineering")
    parser.add_argument(
        "--corrections",
        default="data/fe_record_corrections.csv",
        help="Literature-verified DOI/product FE and voltage corrections",
    )
    args = parser.parse_args()

    # Step 1: Load gold corpus
    print(f"Loading gold corpus from {args.gold_corpus} ...")
    gold_df = pd.read_excel(args.gold_corpus)
    print(f"  {len(gold_df)} rows, {gold_df['doi'].nunique()} papers")

    # Step 2: Extract structured records
    print("Extracting structured records ...")
    records = extract_records_from_gold(gold_df)
    records_df = pd.DataFrame(records)
    records_df = apply_record_corrections(records_df, args.corrections)
    print(f"  {len(records_df)} records from {records_df['doi'].nunique()} papers")

    if records_df.empty:
        print("No records extracted. Check gold_corpus format.")
        return

    # Save intermediate records
    Path(args.records_output).parent.mkdir(parents=True, exist_ok=True)
    records_df.to_csv(args.records_output, index=False)
    print(f"  Saved structured records → {args.records_output}")

    # Summary statistics
    print(f"\n--- Record Summary ---")
    print(f"  Catalyst types: {records_df['catalyst_type'].value_counts().to_dict()}")
    print(f"  Products: {records_df['product'].value_counts().to_dict()}")
    print(f"  FE% range: {records_df['fe_pct'].min():.1f} - {records_df['fe_pct'].max():.1f}")
    print(f"  FE% mean: {records_df['fe_pct'].mean():.1f}")
    print(f"  Voltage available: {records_df['voltage_V'].notna().sum()} / {len(records_df)}")
    print(f"  Electrolyte: {records_df['electrolyte'].value_counts().to_dict()}")

    # Step 3: Build feature matrix
    print("\nBuilding feature matrix ...")
    featured_df = build_feature_matrix(records_df)

    # Fill missing voltage with median
    if featured_df["voltage_V"].notna().any():
        median_v = featured_df["voltage_V"].median()
        featured_df["voltage_V"] = featured_df["voltage_V"].fillna(median_v)
        print(f"  Filled {featured_df['voltage_V'].isna().sum()} missing voltages with median={median_v:.2f}")

    feature_cols = get_feature_columns(featured_df)
    print(f"  Total features: {len(feature_cols)}")

    # Step 4: Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(args.output, index=False)
    print(f"\nSaved ML dataset ({len(featured_df)} rows, {len(featured_df.columns)} cols) → {args.output}")

    # Feature summary
    print(f"\n--- Feature Categories ---")
    magpie_cols = [c for c in feature_cols if c.startswith("magpie_")]
    elem_cols = [c for c in feature_cols if c.startswith("elem_")]
    cat_cols = [c for c in feature_cols if c.startswith("cattype_")]
    prod_cols = [c for c in feature_cols if c.startswith("prod_")]
    elyte_cols = [c for c in feature_cols if c.startswith("elyte_")]
    print(f"  Magpie stats: {len(magpie_cols)}")
    print(f"  Element fractions: {len(elem_cols)}")
    print(f"  Catalyst type one-hot: {len(cat_cols)}")
    print(f"  Product one-hot: {len(prod_cols)}")
    print(f"  Electrolyte one-hot: {len(elyte_cols)}")
    print(f"  Other (voltage, cell, composition stats): {len(feature_cols) - len(magpie_cols) - len(elem_cols) - len(cat_cols) - len(prod_cols) - len(elyte_cols)}")


if __name__ == "__main__":
    main()
