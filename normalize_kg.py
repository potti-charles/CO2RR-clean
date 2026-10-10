"""
normalize_kg.py
===============
Normalize knowledge graph entities: merge synonyms for products and catalysts,
rebuild KG with clean nodes/edges, generate leaderboard and statistics.

Usage:
    python normalize_kg.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
INPUT_FILE = REPO_ROOT / "relation_extraction_results" / "relation_results.jsonl"
OUTPUT_DIR = REPO_ROOT / "relation_extraction_results"

# ---------------------------------------------------------------------------
# Product normalization mapping
# ---------------------------------------------------------------------------
PRODUCT_SYNONYMS = {
    # C1 products
    "CO": ["CO", "carbon monoxide"],
    "HCOOH": ["HCOOH", "formate", "formic acid", "HCOO-", "HCOO(-)", "HCOO"],
    "CH4": ["CH4", "methane"],
    "CH3OH": ["CH3OH", "methanol", "MeOH"],
    "HCHO": ["HCHO", "formaldehyde"],
    "CH3COOH": ["CH3COOH", "acetic acid", "acetate", "CH3COO-"],
    # C2 products
    "C2H4": ["C2H4", "ethylene", "ethene", "ethene (C2H4)", "ethene (ethylene)"],
    "C2H5OH": ["C2H5OH", "ethanol", "EtOH", "C2H6O", "ethanol (CH3CH2OH)", "C2H5 OH"],
    "C2H6": ["C2H6", "ethane", "ethane (C2H6)"],
    "CH3CHO": ["CH3CHO", "acetaldehyde"],
    "CH3COCH3": ["acetone"],
    "ethyl acetate": ["ethyl acetate"],
    # C3+ products
    # C3+ alcohols — keep isomer ambiguity explicit.
    # Only unambiguously *named* isomers map to n-propanol / isopropanol.
    # Bare "C3H7OH" / "propanol" retain unspecified-isomer status so they are
    # NOT silently promoted to n-propanol (which would bias C3 rankings).
    "n-propanol": ["n-propanol", "1-propanol", "n-PrOH",
                    "n-propanol (CH3CH2CH2OH)", "N-propanol", "n-Propanol"],
    "isopropanol": ["isopropanol", "2-propanol", "isopropyl alcohol", "i-PrOH"],
    "propanol": ["C3H7OH", "propanol", "C3H8O"],  # unspecified isomer
    "n-butanol": ["n-butanol"],
    "propylene": ["propylene", "C3H6"],
    # Gas
    "H2": ["H2", "H-2", "hydrogen", "hydrogen evolution"],
    "syngas": ["syngas", "H2 and CO (syngas)", "H2/CO", "CO/H2", "CO and H2", "CO and H-2"],
}

# Build reverse lookup: synonym -> canonical name
_PRODUCT_LOOKUP = {}
for canonical, synonyms in PRODUCT_SYNONYMS.items():
    for syn in synonyms:
        _PRODUCT_LOOKUP[syn.lower().strip()] = canonical

# Group labels (not individual products)
GROUP_LABELS = {
    "C2+", "C2+ products", "C2", "C-2 products", "C2 products",
    "C-2 product", "C-2 reduction products", "C-2 oxygenates",
    "C-2 hydrocarbons", "C2+ alcohols", "C2+ fuels",
    "total C-2 products", "C2 hydrocarbons (ethylene and ethanol)",
    "C-1 products", "C1 products", "total C1 products", "C1",
    "C2+3 products", "C1-C2 products",
    "multicarbon products", "multicarbon",
    "hydrocarbons", "liquid products", "multiple products",
    "oxygenates", "low-carbon alcohols",
}

# Multi-product strings that should be split
MULTI_PRODUCT_PATTERNS = [
    ("methanol, ethanol, isopropanol", ["CH3OH", "C2H5OH", "isopropanol"]),
    ("ethylene and ethanol", ["C2H4", "C2H5OH"]),
    ("ethanol/ethylene", ["C2H5OH", "C2H4"]),
    ("CO + formate", ["CO", "HCOOH"]),
    ("CO, HCOO-", ["CO", "HCOOH"]),
    ("C2H4, C2H6", ["C2H4", "C2H6"]),
    ("C2H4, C2H5OH, C3H7OH", ["C2H4", "C2H5OH", "propanol"]),
    ("C2H4 and C2H5OH", ["C2H4", "C2H5OH"]),
    ("C2H4, C2H6, C2H5OH", ["C2H4", "C2H6", "C2H5OH"]),
    ("CO, CH4 and C2H4", ["CO", "CH4", "C2H4"]),
    ("ethylene, ethanol and n-propanol", ["C2H4", "C2H5OH", "n-propanol"]),
]


def normalize_product(raw: str) -> list[str]:
    """Normalize a product name. Returns list (may split multi-product strings)."""
    raw = raw.strip()

    # Check multi-product patterns first
    for pattern, products in MULTI_PRODUCT_PATTERNS:
        if raw.lower() == pattern.lower():
            return products

    # Check direct lookup
    key = raw.lower().strip()
    if key in _PRODUCT_LOOKUP:
        return [_PRODUCT_LOOKUP[key]]

    # Check if it contains a known product in parentheses: "ethene (C2H4)" -> C2H4
    paren_match = re.search(r'\(([^)]+)\)', raw)
    if paren_match:
        inner = paren_match.group(1).lower().strip()
        if inner in _PRODUCT_LOOKUP:
            return [_PRODUCT_LOOKUP[inner]]

    # Check if it's a group label (C2+, multicarbon, etc.)
    if raw in GROUP_LABELS:
        return [raw]  # Keep as-is, these are category labels

    # Fallback: return as-is
    return [raw]


# ---------------------------------------------------------------------------
# Catalyst normalization
# ---------------------------------------------------------------------------
def normalize_catalyst(raw: str) -> str:
    """Normalize catalyst name: remove common descriptors, standardize format."""
    s = raw.strip()

    # Remove trailing descriptions in parentheses if they're size/morphology
    s = re.sub(r'\s*\((?:NPs?|nanoparticles|nanosheets|nanocubes|nanowires|nanorods|nanofibers|nanodendrites)\)\s*$', '', s, flags=re.IGNORECASE)

    # Remove common prefixes that are morphology descriptors
    prefixes_to_strip = [
        r'^nanostructured\s+',
        r'^nanoporous\s+',
        r'^mesoporous\s+',
        r'^hierarchical\s+',
        r'^flower-like\s+',
        r'^ultrathin\s+',
    ]
    for pat in prefixes_to_strip:
        s = re.sub(pat, '', s, flags=re.IGNORECASE)

    # Remove trailing " NPs", " nanoparticles", " nanosheets", etc.
    s = re.sub(r'\s+(NPs?|nanoparticles|nanosheets|nanocubes|nanowires|nanorods|nanofibers|nanodendrites|nanoflowers|nanostructures|nanocrystals|nanoarrays|nanosheet arrays|nanowire arrays)$', '', s, flags=re.IGNORECASE)

    # Standardize common patterns
    s = re.sub(r'\s+', ' ', s).strip()

    # Remove "derived" suffixes that describe synthesis
    # e.g. "Cu2O-derived Cu" -> keep as is (different from Cu2O)

    return s if s else raw.strip()


# ---------------------------------------------------------------------------
# Parse FE value to float
# ---------------------------------------------------------------------------
# Qualifier prefixes that mark an FE as approximate / bounded rather than exact.
# We keep the numeric value but record the qualifier so downstream ranking /
# averaging can treat these records appropriately (they are NOT exact point values).
_FE_QUALIFIER_RE = re.compile(
    r'^\s*(over|greater than|>=|≥|up to|~|approximately|about|less than|<|≤)\s*',
    re.IGNORECASE,
)
_QUALIFIER_NORMALIZE = {
    'over': '>', 'greater than': '>', '>': '>', '>=': '>=', '≥': '>=',
    'up to': '<=', 'less than': '<', '<': '<', '≤': '<=',
    '~': '~', 'approximately': '~', 'about': '~',
}


def parse_fe_qualifier(fe_str) -> str | None:
    """Return a normalized qualifier ('>','>=','<=','<','~', or None for exact)."""
    if not fe_str or fe_str in ('null', 'N/A', None):
        return None
    m = _FE_QUALIFIER_RE.search(str(fe_str))
    if m:
        return _QUALIFIER_NORMALIZE.get(m.group(1).lower(), m.group(1).lower())
    return None


def parse_fe_value(fe_str) -> float | None:
    """Parse FE string to float percentage (qualifier is captured separately)."""
    if not fe_str or fe_str in ('null', 'N/A', None):
        return None
    fe_str = str(fe_str)
    # Strip qualifier prefix before extracting the number, but parse_fe_qualifier
    # has already recorded it so we do NOT lose the information.
    fe_str = _FE_QUALIFIER_RE.sub('', fe_str)
    m = re.search(r'([\d.]+)\s*%', fe_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def parse_potential(pot_str) -> float | None:
    """Parse potential string to float voltage."""
    if not pot_str or pot_str in ('null', 'N/A', None):
        return None
    pot_str = str(pot_str)
    m = re.search(r'(-?[\d.]+)\s*V', pot_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Main normalization pipeline
# ---------------------------------------------------------------------------
def main():
    # Load results
    results = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"Loaded {len(results)} papers")

    # Normalize all relations
    normalized_relations = []
    skipped_group = 0

    for paper in results:
        source = paper.get("source", "")
        for rel in paper.get("relations", []):
            catalyst_raw = rel["catalyst"]
            product_raw = rel["product"]
            fe_raw = rel.get("faradaic_efficiency")
            pot_raw = rel.get("potential")
            cd_raw = rel.get("current_density")
            elec_raw = rel.get("electrolyte")

            catalyst_norm = normalize_catalyst(catalyst_raw)
            products_norm = normalize_product(product_raw)

            fe_val = parse_fe_value(fe_raw)
            fe_qual = parse_fe_qualifier(fe_raw)
            pot_val = parse_potential(pot_raw)

            # If one raw relation yielded several products, a single FE value
            # almost certainly refers to a combined selectivity, not to each
            # product individually. Flag these as "shared" so per-product
            # ranking / averaging can exclude them (they would otherwise inflate
            # catalyst leaderboards — e.g. a 10.7% "liquid products" FE wrongly
            # applied to CH3OH, C2H5OH AND isopropanol at once).
            is_shared_fe = len(products_norm) > 1 and fe_val is not None

            for prod in products_norm:
                # Skip vague group labels for the leaderboard
                is_group = prod in GROUP_LABELS
                if is_group:
                    skipped_group += 1

                normalized_relations.append({
                    "source": source,
                    "catalyst_raw": catalyst_raw,
                    "catalyst": catalyst_norm,
                    "product_raw": product_raw,
                    "product": prod,
                    "is_group_label": is_group,
                    "fe_raw": fe_raw,
                    "fe_pct": fe_val,
                    "fe_qualifier": fe_qual,
                    "fe_scope": "shared" if is_shared_fe else "single",
                    "potential_raw": pot_raw,
                    "potential_V": pot_val,
                    "current_density": cd_raw,
                    "electrolyte": elec_raw,
                })

    print(f"Normalized relations: {len(normalized_relations)} (from {sum(r['num_relations'] for r in results)} raw)")
    print(f"Group labels skipped for leaderboard: {skipped_group}")

    # Save normalized relations
    norm_df = pd.DataFrame(normalized_relations)
    norm_df.to_csv(OUTPUT_DIR / "normalized_relations.csv", index=False)
    print(f"Saved → normalized_relations.csv")

    # --- Product distribution ---
    specific_rels = norm_df[~norm_df["is_group_label"]]
    prod_counts = specific_rels["product"].value_counts()
    print(f"\n=== Product Distribution (after normalization) ===")
    print(f"Unique products: {len(prod_counts)}")
    for prod, cnt in prod_counts.head(15).items():
        print(f"  {prod:15s}: {cnt}")

    # --- Catalyst distribution ---
    cat_counts = specific_rels["catalyst"].value_counts()
    print(f"\n=== Catalyst Distribution (after normalization) ===")
    print(f"Unique catalysts: {len(cat_counts)}")
    for cat, cnt in cat_counts.head(15).items():
        print(f"  {cat:30s}: {cnt}")

    # --- Leaderboard: best catalyst per product by FE ---
    # Exclude "shared" FE records: their FE is a combined selectivity for several
    # products and must NOT be attributed to any single product in a ranking.
    with_fe = specific_rels[
        specific_rels["fe_pct"].notna() & (specific_rels.get("fe_scope") != "shared")
    ].copy()
    print(f"\n=== Catalyst Leaderboard (by FE) ===")
    print(f"Relations with FE values (excl. shared): {len(with_fe)}")

    leaderboard = []
    for prod in with_fe["product"].unique():
        prod_data = with_fe[with_fe["product"] == prod].sort_values("fe_pct", ascending=False)
        best = prod_data.iloc[0]
        leaderboard.append({
            "product": prod,
            "best_catalyst": best["catalyst"],
            "best_fe_pct": best["fe_pct"],
            "potential": best["potential_raw"],
            "source": best["source"],
            "total_entries": len(prod_data),
            "avg_fe_pct": round(prod_data["fe_pct"].mean(), 1),
        })

    lb_df = pd.DataFrame(leaderboard).sort_values("best_fe_pct", ascending=False)
    lb_df.to_csv(OUTPUT_DIR / "catalyst_leaderboard.csv", index=False)

    print(f"\nTop catalysts per product:")
    print(f"{'Product':15s} {'Best Catalyst':35s} {'FE%':>8s} {'Potential':>20s} {'#Entries':>8s}")
    print("-" * 90)
    for _, row in lb_df.iterrows():
        print(f"{row['product']:15s} {row['best_catalyst']:35s} {row['best_fe_pct']:7.1f}% {str(row['potential'] or ''):>20s} {row['total_entries']:8d}")

    # --- Build normalized KG ---
    print(f"\n=== Building Normalized Knowledge Graph ===")
    nodes = {}
    edges = []
    for _, rel in specific_rels.iterrows():
        cat = rel["catalyst"].lower()
        prod = rel["product"]

        if cat not in nodes:
            nodes[cat] = {"id": cat, "type": "catalyst", "label": rel["catalyst"], "count": 0}
        nodes[cat]["count"] += 1

        prod_key = f"product_{prod}"
        if prod_key not in nodes:
            nodes[prod_key] = {"id": prod_key, "type": "product", "label": prod, "count": 0}
        nodes[prod_key]["count"] += 1

        edges.append({
            "source": cat,
            "target": prod_key,
            "fe_pct": rel["fe_pct"],
            "potential_V": rel["potential_V"],
            "paper": rel["source"],
        })

    # Export to GraphML
    try:
        import networkx as nx
        G = nx.DiGraph()
        for node in nodes.values():
            G.add_node(node["id"], label=node["label"], node_type=node["type"],
                       mention_count=node["count"])
        for edge in edges:
            fe_label = f"FE={edge['fe_pct']:.1f}%" if edge["fe_pct"] else ""
            G.add_edge(edge["source"], edge["target"],
                       fe_pct=edge["fe_pct"] or 0,
                       potential_V=edge["potential_V"] or 0,
                       label=fe_label,
                       paper=edge["paper"])
        graphml_path = OUTPUT_DIR / "co2rr_knowledge_graph_normalized.graphml"
        nx.write_graphml(G, str(graphml_path))
        print(f"GraphML → {graphml_path}")
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    except ImportError:
        print("networkx not available, skipping GraphML export")

    # --- Catalyst-Product matrix (heatmap data) ---
    top_cats = cat_counts.head(20).index.tolist()
    top_prods = ["CO", "C2H4", "HCOOH", "CH4", "C2H5OH", "CH3OH", "H2", "C2H6",
                 "n-propanol", "CH3COOH"]

    matrix_data = []
    for cat in top_cats:
        cat_rels = with_fe[with_fe["catalyst"].str.lower() == cat.lower()]
        row = {"catalyst": cat}
        for prod in top_prods:
            prod_rels = cat_rels[cat_rels["product"] == prod]
            if len(prod_rels) > 0:
                row[prod] = round(prod_rels["fe_pct"].max(), 1)
            else:
                row[prod] = None
        matrix_data.append(row)

    matrix_df = pd.DataFrame(matrix_data)
    matrix_df.to_csv(OUTPUT_DIR / "catalyst_product_matrix.csv", index=False)
    print(f"\nCatalyst-Product FE matrix saved → catalyst_product_matrix.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
