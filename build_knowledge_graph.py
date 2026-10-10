"""
build_knowledge_graph.py
========================
Build a proper CO2RR knowledge graph with:
  - Domain ontology (catalyst type hierarchy, multiple relation types)
  - Entity disambiguation (unique URIs, catalyst type classification)
  - Standard graph formats (GraphML, RDF/Turtle, Neo4j Cypher)
  - Graph query interface (networkx + example Cypher queries)

Usage:
    python build_knowledge_graph.py                    # Build KG
    python build_knowledge_graph.py --query            # Interactive query mode
    python build_knowledge_graph.py --stats            # Print KG statistics
    python build_knowledge_graph.py --export-neo4j     # Export Neo4j import script
"""

import argparse
import html
import json
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
INPUT_FILE = REPO_ROOT / "relation_extraction_results" / "normalized_relations.csv"
OUTPUT_DIR = REPO_ROOT / "relation_extraction_results" / "knowledge_graph"
TITLE_LOOKUP_FILE = REPO_ROOT / "titles_dois.csv"


def normalize_paper_title(title: str) -> str:
    text = html.unescape(str(title))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def load_paper_title_lookup(csv_path: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    exact_titles: dict[str, str] = {}
    ordered_titles: list[str] = []
    if not csv_path.exists():
        return exact_titles, tuple(ordered_titles)

    try:
        df = pd.read_csv(csv_path, usecols=["query_title"])
    except ValueError:
        return exact_titles, tuple(ordered_titles)

    for raw_title in df["query_title"].dropna().astype(str):
        title = normalize_paper_title(raw_title)
        if not title or title.lower() == "nan":
            continue
        if title in exact_titles:
            continue
        exact_titles[title] = title
        ordered_titles.append(title)

    return exact_titles, tuple(ordered_titles)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CO2RR DOMAIN ONTOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

ONTOLOGY = {
    # --- Node types (classes) ---
    "classes": {
        "Catalyst": {
            "description": "Electrocatalyst material for CO2 reduction",
            "subclasses": {
                "MetalCatalyst": "Pure metal or metal foil catalyst",
                "AlloyCatalyst": "Bimetallic or multi-metallic alloy",
                "MetalOxideCatalyst": "Metal oxide catalyst",
                "CarbonBasedCatalyst": "Carbon-based or heteroatom-doped carbon",
                "MOFCatalyst": "Metal-organic framework based catalyst",
                "SAC": "Single-atom catalyst",
                "CompositeCatalyst": "Composite or supported catalyst",
                "OtherCatalyst": "Other catalyst types",
            },
        },
        "Product": {
            "description": "CO2 reduction reaction product",
            "subclasses": {
                "C1Product": "Single-carbon product (CO, HCOOH, CH4, CH3OH, HCHO)",
                "C2Product": "Two-carbon product (C2H4, C2H5OH, C2H6, CH3CHO, CH3COOH)",
                "C3PlusProduct": "Three or more carbon product (n-propanol, propylene)",
                "HER": "Hydrogen evolution reaction product (H2)",
            },
        },
        "Electrolyte": {
            "description": "Electrolyte solution used in CO2RR",
        },
        "Paper": {
            "description": "Scientific publication",
        },
    },

    # --- Relation types (edge types) ---
    "relations": {
        "PRODUCES": {
            "domain": "Catalyst",
            "range": "Product",
            "properties": ["faradaic_efficiency", "potential", "current_density"],
            "description": "Catalyst produces this product via CO2RR",
        },
        "REPORTED_IN": {
            "domain": "Catalyst",
            "range": "Paper",
            "description": "Catalyst performance reported in this paper",
        },
        "USES_ELECTROLYTE": {
            "domain": "Catalyst",
            "range": "Electrolyte",
            "description": "Catalyst tested in this electrolyte",
        },
        "IS_A": {
            "domain": "Catalyst",
            "range": "CatalystType",
            "description": "Catalyst belongs to this type category",
        },
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CATALYST TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns for classifying catalysts into ontology types
CATALYST_TYPE_RULES = [
    # SAC (single atom) — check first, most specific
    (r"single.?atom|SA[Cs]|\bSA\b|single.?site|isolated\s+(metal\s+)?atom",
     "SAC", re.IGNORECASE),
    # MOF
    (r"\bMOF\b|metal.?organic.?framework|ZIF-|UiO-|HKUST|MIL-|PCN-",
     "MOFCatalyst", re.IGNORECASE),
    # Carbon-based (N-doped carbon, graphene, CNT, etc.)
    (r"\b(N-C|NC|N-doped|graphene|CNT|carbon\s+nano|g-C3N4|porphyrin|phthalocyanine"
     r"|metal-?N-?C|M-?N-?C|Fe-?N-?C|Co-?N-?C|Ni-?N-?C|Cu-?N-?C|Mn-?N-?C"
     r"|nitrogen.?doped.?carbon|N,\s*\w+-doped)\b",
     "CarbonBasedCatalyst", re.IGNORECASE),
    # Alloy (bimetallic patterns)
    (r"(Cu-?[A-Z][a-z]|[A-Z][a-z]-?Cu|Au-?Ag|Ag-?Au|Pd-?[A-Z]|[A-Z][a-z]-?Pd"
     r"|alloy|bimetal|CuAu|AuCu|CuAg|AgCu|CuSn|SnCu|CuZn|ZnCu|CuIn|InCu"
     r"|CuBi|BiCu|PdCu|CuPd|AuPd|PdAu|CuNi|NiCu|CuFe|FeCu|AgSn|SnAg"
     r"|Cu\d*[A-Z][a-z]\d*|[A-Z][a-z]\d*Cu\d*)\b",
     "AlloyCatalyst", re.IGNORECASE),
    # Metal oxide
    (r"(oxide|O[xX]|SnO|CuO|Cu2O|ZnO|In2O3|Bi2O3|TiO2|CoO|FeO|MnO|NiO"
     r"|Co3O4|Fe2O3|Fe3O4|CeO2|RuO2|IrO2|PbO2|Pb3O4|Ga2O3|Ag2O)\b",
     "MetalOxideCatalyst", re.IGNORECASE),
    # Composite/supported
    (r"(supported|@|/|on\s+|loaded|decorated|anchored|deposited|coated|core.?shell"
     r"|heterostructure|composite|hybrid|GDE|gas\s+diffusion)",
     "CompositeCatalyst", re.IGNORECASE),
    # Pure metal
    (r"^(Cu|Ag|Au|Sn|Bi|Pb|Zn|In|Pd|Pt|Ni|Co|Fe|Mn|Mo|W|Ti|Ga|Cd|Hg|Ru|Ir|Re|Rh)$",
     "MetalCatalyst", re.IGNORECASE),
]

# Product type mapping
PRODUCT_TYPE_MAP = {
    "CO": "C1Product", "HCOOH": "C1Product", "CH4": "C1Product",
    "CH3OH": "C1Product", "HCHO": "C1Product",
    "C2H4": "C2Product", "C2H5OH": "C2Product", "C2H6": "C2Product",
    "CH3CHO": "C2Product", "CH3COOH": "C2Product", "CH3COCH3": "C2Product",
    "ethyl acetate": "C2Product",
    "n-propanol": "C3PlusProduct", "isopropanol": "C3PlusProduct",
    "n-butanol": "C3PlusProduct", "propylene": "C3PlusProduct",
    "H2": "HER", "syngas": "C1Product",
}

# Electrolyte normalization
ELECTROLYTE_SYNONYMS = {
    "0.1 M KHCO3": ["0.1 M KHCO3", "0.1M KHCO3", "0.1 mol/L KHCO3",
                     "0.1 M potassium bicarbonate"],
    "0.5 M KHCO3": ["0.5 M KHCO3", "0.5M KHCO3", "0.5 mol/L KHCO3",
                     "0.5 M potassium bicarbonate"],
    "1.0 M KOH": ["1 M KOH", "1.0 M KOH", "1M KOH",
                   "1 M potassium hydroxide"],
    "0.1 M KOH": ["0.1 M KOH", "0.1M KOH"],
    "0.5 M H2SO4": ["0.5 M H2SO4", "0.5M H2SO4"],
    "0.1 M CsHCO3": ["0.1 M CsHCO3", "0.1M CsHCO3"],
    "0.5 M CsHCO3": ["0.5 M CsHCO3"],
    "[BMIM]BF4": ["[BMIM]BF4", "BMIMBF4", "ionic liquid"],
}

_ELEC_LOOKUP = {}
for canonical, syns in ELECTROLYTE_SYNONYMS.items():
    for s in syns:
        _ELEC_LOOKUP[s.lower().strip()] = canonical


def classify_catalyst(name: str) -> str:
    """Classify catalyst into ontology type."""
    for pattern, cat_type, flags in CATALYST_TYPE_RULES:
        if re.search(pattern, name, flags):
            return cat_type
    return "OtherCatalyst"


def make_uri(entity_type: str, name: str) -> str:
    """Generate a unique URI for an entity."""
    # Use hash for stable, collision-free IDs
    slug = re.sub(r'[^a-zA-Z0-9]', '_', name.strip().lower())
    short_hash = hashlib.md5(name.strip().lower().encode()).hexdigest()[:8]
    return f"co2rr:{entity_type}/{slug}_{short_hash}"


def normalize_electrolyte(raw: str) -> str | None:
    """Normalize electrolyte string."""
    if not raw or raw in ('null', 'N/A', 'None', None):
        return None
    key = str(raw).lower().strip()
    return _ELEC_LOOKUP.get(key, str(raw).strip())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KNOWLEDGE GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class CO2RRKnowledgeGraph:
    """CO2RR domain knowledge graph with ontology, entities, and relations."""

    def __init__(self):
        self.G = nx.MultiDiGraph()  # MultiDiGraph supports parallel edges
        self.entity_index = {}  # name -> URI
        self.stats = Counter()
        self.paper_titles_exact, self.paper_titles_ordered = load_paper_title_lookup(
            TITLE_LOOKUP_FILE
        )

    def _resolve_paper_title(self, title: str) -> str:
        normalized = normalize_paper_title(title)
        if not normalized or normalized.lower() == "nan":
            return normalized
        resolved = self.paper_titles_exact.get(normalized)
        if resolved:
            return resolved

        prefix_matches = [full_title for full_title in self.paper_titles_ordered
                          if full_title.startswith(normalized)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        return normalized

    def build_from_csv(self, csv_path: str | Path):
        """Build KG from normalized_relations.csv."""
        df = pd.read_csv(csv_path)
        df = df[df["is_group_label"] == False].copy()
        print(f"Loading {len(df)} relations from {csv_path}")

        for _, row in df.iterrows():
            catalyst = str(row["catalyst"]).strip()
            product = str(row["product"]).strip()
            paper = self._resolve_paper_title(str(row["source"]).strip())
            fe_raw = row.get("fe_raw")
            fe_pct = row.get("fe_pct")
            potential_raw = row.get("potential_raw")
            potential_v = row.get("potential_V")
            current_density = row.get("current_density")
            electrolyte_raw = row.get("electrolyte")

            # --- Add Catalyst node ---
            cat_uri = self._add_catalyst(catalyst)

            # --- Add Product node ---
            prod_uri = self._add_product(product)

            # --- Add Paper node ---
            paper_uri = self._add_paper(paper)

            # --- Add Electrolyte node ---
            elec_norm = normalize_electrolyte(electrolyte_raw)
            elec_uri = None
            if elec_norm:
                elec_uri = self._add_electrolyte(elec_norm)

            # --- Add PRODUCES edge ---
            edge_props = {
                "relation_type": "PRODUCES",
                "paper": paper,
            }
            if pd.notna(fe_pct):
                edge_props["faradaic_efficiency"] = float(fe_pct)
            if fe_raw and str(fe_raw) not in ('nan', 'None'):
                edge_props["fe_raw"] = str(fe_raw)
            if pd.notna(potential_v):
                edge_props["potential_V"] = float(potential_v)
            if potential_raw and str(potential_raw) not in ('nan', 'None'):
                edge_props["potential_raw"] = str(potential_raw)
            if current_density and str(current_density) not in ('nan', 'None'):
                edge_props["current_density"] = str(current_density)

            self.G.add_edge(cat_uri, prod_uri, **edge_props)
            self.stats["edges_produces"] += 1

            # --- Add REPORTED_IN edge ---
            self.G.add_edge(cat_uri, paper_uri, relation_type="REPORTED_IN")
            self.stats["edges_reported_in"] += 1

            # --- Add USES_ELECTROLYTE edge ---
            if elec_uri:
                self.G.add_edge(cat_uri, elec_uri, relation_type="USES_ELECTROLYTE")
                self.stats["edges_uses_electrolyte"] += 1

        print(f"\nKnowledge Graph Built:")
        print(f"  Nodes: {self.G.number_of_nodes()}")
        print(f"  Edges: {self.G.number_of_edges()}")
        for k, v in sorted(self.stats.items()):
            print(f"  {k}: {v}")

    def _add_catalyst(self, name: str) -> str:
        key = name.lower().strip()
        if key in self.entity_index:
            return self.entity_index[key]

        uri = make_uri("catalyst", name)
        cat_type = classify_catalyst(name)
        self.G.add_node(uri,
                        entity_type="Catalyst",
                        catalyst_type=cat_type,
                        label=name,
                        uri=uri)
        self.entity_index[key] = uri
        self.stats[f"catalyst_{cat_type}"] += 1
        self.stats["nodes_catalyst"] += 1
        return uri

    def _add_product(self, name: str) -> str:
        key = f"product_{name.lower().strip()}"
        if key in self.entity_index:
            return self.entity_index[key]

        uri = make_uri("product", name)
        prod_type = PRODUCT_TYPE_MAP.get(name, "OtherProduct")
        self.G.add_node(uri,
                        entity_type="Product",
                        product_type=prod_type,
                        label=name,
                        uri=uri)
        self.entity_index[key] = uri
        self.stats[f"product_{prod_type}"] += 1
        self.stats["nodes_product"] += 1
        return uri

    def _add_paper(self, title: str) -> str:
        key = f"paper_{title.lower().strip()}"
        if key in self.entity_index:
            return self.entity_index[key]

        uri = make_uri("paper", title)
        self.G.add_node(uri,
                        entity_type="Paper",
                        label=title[:100],
                        full_title=title,
                        uri=uri)
        self.entity_index[key] = uri
        self.stats["nodes_paper"] += 1
        return uri

    def _add_electrolyte(self, name: str) -> str:
        key = f"electrolyte_{name.lower().strip()}"
        if key in self.entity_index:
            return self.entity_index[key]

        uri = make_uri("electrolyte", name)
        self.G.add_node(uri,
                        entity_type="Electrolyte",
                        label=name,
                        uri=uri)
        self.entity_index[key] = uri
        self.stats["nodes_electrolyte"] += 1
        return uri

    # ───────────────────────────────────────────────────────────────────────
    # EXPORT METHODS
    # ───────────────────────────────────────────────────────────────────────

    def export_graphml(self, path: Path):
        """Export to GraphML format (compatible with Gephi, Cytoscape, etc.)."""
        # Convert MultiDiGraph to DiGraph for GraphML (merge parallel edges)
        G_simple = nx.DiGraph()
        for node, data in self.G.nodes(data=True):
            G_simple.add_node(node, **{k: str(v) for k, v in data.items()})
        for u, v, data in self.G.edges(data=True):
            key = f"{u}->{v}:{data.get('relation_type', '')}"
            if not G_simple.has_edge(u, v):
                G_simple.add_edge(u, v, **{k: str(v_) for k, v_ in data.items()})
        nx.write_graphml(G_simple, str(path))
        print(f"GraphML exported → {path}")

    def export_rdf_turtle(self, path: Path):
        """Export to RDF Turtle format — standard knowledge graph format."""
        lines = []
        lines.append("@prefix co2rr: <http://co2rr.org/ontology/> .")
        lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
        lines.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
        lines.append("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
        lines.append("")

        # --- Ontology definitions ---
        lines.append("# === Ontology Classes ===")
        for cls, info in ONTOLOGY["classes"].items():
            lines.append(f'co2rr:{cls} rdf:type rdfs:Class ;')
            lines.append(f'    rdfs:label "{cls}" ;')
            lines.append(f'    rdfs:comment "{info["description"]}" .')
            if "subclasses" in info:
                for sub, desc in info["subclasses"].items():
                    lines.append(f'co2rr:{sub} rdf:type rdfs:Class ;')
                    lines.append(f'    rdfs:subClassOf co2rr:{cls} ;')
                    lines.append(f'    rdfs:label "{sub}" ;')
                    lines.append(f'    rdfs:comment "{desc}" .')
        lines.append("")

        lines.append("# === Ontology Relations ===")
        for rel, info in ONTOLOGY["relations"].items():
            lines.append(f'co2rr:{rel} rdf:type rdf:Property ;')
            lines.append(f'    rdfs:label "{rel}" ;')
            lines.append(f'    rdfs:comment "{info["description"]}" .')
        lines.append("")

        # --- Entity instances ---
        lines.append("# === Entity Instances ===")
        for node, data in self.G.nodes(data=True):
            etype = data.get("entity_type", "Thing")
            label = data.get("label", node).replace('"', '\\"')
            uri_safe = node.replace(" ", "_")

            if etype == "Catalyst":
                cat_type = data.get("catalyst_type", "OtherCatalyst")
                lines.append(f'<{uri_safe}> rdf:type co2rr:{cat_type} ;')
                lines.append(f'    rdfs:label "{label}" .')
            elif etype == "Product":
                prod_type = data.get("product_type", "Product")
                lines.append(f'<{uri_safe}> rdf:type co2rr:{prod_type} ;')
                lines.append(f'    rdfs:label "{label}" .')
            elif etype == "Electrolyte":
                lines.append(f'<{uri_safe}> rdf:type co2rr:Electrolyte ;')
                lines.append(f'    rdfs:label "{label}" .')
            elif etype == "Paper":
                lines.append(f'<{uri_safe}> rdf:type co2rr:Paper ;')
                lines.append(f'    rdfs:label "{label}" .')
        lines.append("")

        # --- Edges ---
        lines.append("# === Relations ===")
        seen_edges = set()
        for u, v, data in self.G.edges(data=True):
            rel_type = data.get("relation_type", "RELATED_TO")
            edge_key = f"{u}|{v}|{rel_type}"
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            u_safe = u.replace(" ", "_")
            v_safe = v.replace(" ", "_")
            lines.append(f'<{u_safe}> co2rr:{rel_type} <{v_safe}> .')

            if rel_type == "PRODUCES" and "faradaic_efficiency" in data:
                fe = data["faradaic_efficiency"]
                lines.append(f'  # FE = {fe}%')

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"RDF Turtle exported → {path}")

    def export_neo4j_cypher(self, path: Path):
        """Export Neo4j Cypher import script."""
        lines = []
        lines.append("// ═══════════════════════════════════════════════════════════════")
        lines.append("// CO2RR Knowledge Graph — Neo4j Import Script")
        lines.append("// Generated by build_knowledge_graph.py")
        lines.append("// ═══════════════════════════════════════════════════════════════")
        lines.append("")

        # --- Constraints & Indexes ---
        lines.append("// --- Schema constraints ---")
        lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Catalyst) REQUIRE c.uri IS UNIQUE;")
        lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.uri IS UNIQUE;")
        lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Electrolyte) REQUIRE e.uri IS UNIQUE;")
        lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.uri IS UNIQUE;")
        lines.append("")
        lines.append("CREATE INDEX IF NOT EXISTS FOR (c:Catalyst) ON (c.name);")
        lines.append("CREATE INDEX IF NOT EXISTS FOR (p:Product) ON (p.name);")
        lines.append("CREATE INDEX IF NOT EXISTS FOR (c:Catalyst) ON (c.catalyst_type);")
        lines.append("")

        # --- Create nodes ---
        lines.append("// --- Catalyst nodes ---")
        catalysts = [(n, d) for n, d in self.G.nodes(data=True)
                     if d.get("entity_type") == "Catalyst"]
        for uri, data in catalysts:
            label = data["label"].replace("'", "\\'")
            cat_type = data.get("catalyst_type", "OtherCatalyst")
            lines.append(
                f"MERGE (c:Catalyst:{cat_type} {{uri: '{uri}'}}) "
                f"SET c.name = '{label}', c.catalyst_type = '{cat_type}';"
            )
        lines.append("")

        lines.append("// --- Product nodes ---")
        products = [(n, d) for n, d in self.G.nodes(data=True)
                    if d.get("entity_type") == "Product"]
        for uri, data in products:
            label = data["label"].replace("'", "\\'")
            prod_type = data.get("product_type", "Product")
            lines.append(
                f"MERGE (p:Product:{prod_type} {{uri: '{uri}'}}) "
                f"SET p.name = '{label}', p.product_type = '{prod_type}';"
            )
        lines.append("")

        lines.append("// --- Electrolyte nodes ---")
        electrolytes = [(n, d) for n, d in self.G.nodes(data=True)
                        if d.get("entity_type") == "Electrolyte"]
        for uri, data in electrolytes:
            label = data["label"].replace("'", "\\'")
            lines.append(
                f"MERGE (e:Electrolyte {{uri: '{uri}'}}) "
                f"SET e.name = '{label}';"
            )
        lines.append("")

        lines.append("// --- Paper nodes ---")
        papers = [(n, d) for n, d in self.G.nodes(data=True)
                  if d.get("entity_type") == "Paper"]
        for uri, data in papers:
            label = data.get("full_title", data["label"]).replace("'", "\\'")
            lines.append(
                f"MERGE (p:Paper {{uri: '{uri}'}}) "
                f"SET p.title = '{label}';"
            )
        lines.append("")

        # --- Create edges ---
        lines.append("// --- PRODUCES edges ---")
        seen = set()
        for u, v, data in self.G.edges(data=True):
            if data.get("relation_type") != "PRODUCES":
                continue
            props = []
            if "faradaic_efficiency" in data:
                props.append(f"fe: {data['faradaic_efficiency']}")
            if "potential_V" in data:
                props.append(f"potential_V: {data['potential_V']}")
            if "potential_raw" in data:
                props.append(f"potential_raw: '{data['potential_raw']}'")
            if "current_density" in data:
                cd = str(data["current_density"]).replace("'", "\\'")
                props.append(f"current_density: '{cd}'")
            if "paper" in data:
                paper = data["paper"][:80].replace("'", "\\'")
                props.append(f"paper: '{paper}'")
            prop_str = ", ".join(props)
            edge_id = f"{u}|{v}|{prop_str}"
            if edge_id in seen:
                continue
            seen.add(edge_id)
            lines.append(
                f"MATCH (c:Catalyst {{uri: '{u}'}}), (p:Product {{uri: '{v}'}}) "
                f"CREATE (c)-[:PRODUCES {{{prop_str}}}]->(p);"
            )
        lines.append("")

        lines.append("// --- REPORTED_IN edges ---")
        seen_ri = set()
        for u, v, data in self.G.edges(data=True):
            if data.get("relation_type") != "REPORTED_IN":
                continue
            key = f"{u}|{v}"
            if key in seen_ri:
                continue
            seen_ri.add(key)
            lines.append(
                f"MATCH (c:Catalyst {{uri: '{u}'}}), (p:Paper {{uri: '{v}'}}) "
                f"MERGE (c)-[:REPORTED_IN]->(p);"
            )
        lines.append("")

        lines.append("// --- USES_ELECTROLYTE edges ---")
        seen_ue = set()
        for u, v, data in self.G.edges(data=True):
            if data.get("relation_type") != "USES_ELECTROLYTE":
                continue
            key = f"{u}|{v}"
            if key in seen_ue:
                continue
            seen_ue.add(key)
            lines.append(
                f"MATCH (c:Catalyst {{uri: '{u}'}}), (e:Electrolyte {{uri: '{v}'}}) "
                f"MERGE (c)-[:USES_ELECTROLYTE]->(e);"
            )
        lines.append("")

        # --- Example queries ---
        lines.append("// ═══════════════════════════════════════════════════════════════")
        lines.append("// EXAMPLE CYPHER QUERIES")
        lines.append("// ═══════════════════════════════════════════════════════════════")
        lines.append("")
        lines.append("// Q1: Top 10 catalysts for CO production by FE")
        lines.append("// MATCH (c:Catalyst)-[r:PRODUCES]->(p:Product {name: 'CO'})")
        lines.append("// WHERE r.fe IS NOT NULL")
        lines.append("// RETURN c.name, r.fe, r.potential_V")
        lines.append("// ORDER BY r.fe DESC LIMIT 10;")
        lines.append("")
        lines.append("// Q2: All products of Cu-based alloy catalysts")
        lines.append("// MATCH (c:AlloyCatalyst)-[:PRODUCES]->(p:Product)")
        lines.append("// WHERE c.name CONTAINS 'Cu'")
        lines.append("// RETURN c.name, p.name, count(*) AS mentions")
        lines.append("// ORDER BY mentions DESC;")
        lines.append("")
        lines.append("// Q3: Which catalysts produce C2+ products with FE > 50%?")
        lines.append("// MATCH (c:Catalyst)-[r:PRODUCES]->(p:C2Product)")
        lines.append("// WHERE r.fe > 50")
        lines.append("// RETURN c.name, p.name, r.fe, r.potential_V")
        lines.append("// ORDER BY r.fe DESC;")
        lines.append("")
        lines.append("// Q4: Papers reporting catalyst X")
        lines.append("// MATCH (c:Catalyst {name: 'Cu'})-[:REPORTED_IN]->(paper:Paper)")
        lines.append("// RETURN paper.title;")
        lines.append("")
        lines.append("// Q5: Catalyst comparison — same product, different catalysts")
        lines.append("// MATCH (c1:Catalyst)-[r1:PRODUCES]->(p:Product {name: 'C2H4'}),")
        lines.append("//       (c2:Catalyst)-[r2:PRODUCES]->(p)")
        lines.append("// WHERE c1 <> c2 AND r1.fe > 30 AND r2.fe > 30")
        lines.append("// RETURN c1.name, r1.fe, c2.name, r2.fe")
        lines.append("// ORDER BY r1.fe DESC LIMIT 20;")
        lines.append("")
        lines.append("// Q6: Knowledge graph path — which catalyst type favors which product?")
        lines.append("// MATCH (c:Catalyst)-[r:PRODUCES]->(p:Product)")
        lines.append("// WHERE r.fe IS NOT NULL")
        lines.append("// RETURN c.catalyst_type, p.name, avg(r.fe) AS avg_fe, count(*) AS n")
        lines.append("// ORDER BY c.catalyst_type, avg_fe DESC;")

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Neo4j Cypher script exported → {path}")

    def export_json_ld(self, path: Path):
        """Export to JSON-LD format."""
        doc = {
            "@context": {
                "co2rr": "http://co2rr.org/ontology/",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "name": "rdfs:label",
                "type": "rdf:type",
                "produces": "co2rr:PRODUCES",
                "reported_in": "co2rr:REPORTED_IN",
                "uses_electrolyte": "co2rr:USES_ELECTROLYTE",
            },
            "@graph": [],
        }

        for node, data in self.G.nodes(data=True):
            entry = {
                "@id": node,
                "@type": f"co2rr:{data.get('entity_type', 'Thing')}",
                "name": data.get("label", ""),
            }
            if data.get("catalyst_type"):
                entry["co2rr:catalyst_type"] = data["catalyst_type"]
            if data.get("product_type"):
                entry["co2rr:product_type"] = data["product_type"]

            # Add outgoing edges
            out_edges = defaultdict(list)
            for _, target, edata in self.G.edges(node, data=True):
                rel = edata.get("relation_type", "RELATED_TO")
                edge_info = {"@id": target}
                if "faradaic_efficiency" in edata:
                    edge_info["co2rr:faradaic_efficiency"] = edata["faradaic_efficiency"]
                if "potential_V" in edata:
                    edge_info["co2rr:potential_V"] = edata["potential_V"]
                out_edges[rel].append(edge_info)

            for rel, targets in out_edges.items():
                entry[f"co2rr:{rel}"] = targets if len(targets) > 1 else targets[0]

            doc["@graph"].append(entry)

        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON-LD exported → {path}")

    # ───────────────────────────────────────────────────────────────────────
    # QUERY METHODS (networkx-based)
    # ───────────────────────────────────────────────────────────────────────

    def query_top_catalysts_for_product(self, product: str, top_n: int = 10):
        """Find top catalysts for a given product by FE (deduplicated, best per catalyst)."""
        prod_nodes = [n for n, d in self.G.nodes(data=True)
                      if d.get("entity_type") == "Product"
                      and d.get("label", "").lower() == product.lower()]
        if not prod_nodes:
            print(f"Product '{product}' not found.")
            return []

        # Collect best FE per catalyst
        best_per_cat = {}  # catalyst_uri -> best record
        for prod_uri in prod_nodes:
            for pred in self.G.predecessors(prod_uri):
                pred_data = self.G.nodes[pred]
                if pred_data.get("entity_type") != "Catalyst":
                    continue
                # Check all edges from pred to prod_uri
                for _, tgt, edata in self.G.edges(pred, data=True):
                    if tgt != prod_uri or edata.get("relation_type") != "PRODUCES":
                        continue
                    fe = edata.get("faradaic_efficiency")
                    if fe is None:
                        continue
                    if pred not in best_per_cat or fe > best_per_cat[pred]["FE%"]:
                        best_per_cat[pred] = {
                            "catalyst": pred_data.get("label", pred),
                            "catalyst_type": pred_data.get("catalyst_type", ""),
                            "FE%": fe,
                            "potential": edata.get("potential_raw", ""),
                            "paper": edata.get("paper", "")[:60],
                        }

        results = sorted(best_per_cat.values(), key=lambda x: x["FE%"], reverse=True)
        return results[:top_n]

    def query_catalyst_products(self, catalyst_name: str):
        """Find all products of a given catalyst."""
        cat_nodes = [n for n, d in self.G.nodes(data=True)
                     if d.get("entity_type") == "Catalyst"
                     and catalyst_name.lower() in d.get("label", "").lower()]
        results = []
        for cat_uri in cat_nodes:
            cat_label = self.G.nodes[cat_uri].get("label", cat_uri)
            for _, target, edata in self.G.edges(cat_uri, data=True):
                if edata.get("relation_type") == "PRODUCES":
                    prod_data = self.G.nodes[target]
                    results.append({
                        "catalyst": cat_label,
                        "product": prod_data.get("label", target),
                        "FE%": edata.get("faradaic_efficiency"),
                        "potential": edata.get("potential_raw", ""),
                    })
        return results

    def query_catalyst_type_stats(self):
        """Statistics by catalyst type."""
        type_stats = defaultdict(lambda: {"count": 0, "products": Counter(), "avg_fe": []})
        for node, data in self.G.nodes(data=True):
            if data.get("entity_type") != "Catalyst":
                continue
            cat_type = data.get("catalyst_type", "Other")
            type_stats[cat_type]["count"] += 1
            for _, target, edata in self.G.edges(node, data=True):
                if edata.get("relation_type") == "PRODUCES":
                    prod = self.G.nodes[target].get("label", "")
                    type_stats[cat_type]["products"][prod] += 1
                    fe = edata.get("faradaic_efficiency")
                    if fe is not None:
                        type_stats[cat_type]["avg_fe"].append(fe)
        return dict(type_stats)

    def print_stats(self):
        """Print comprehensive KG statistics."""
        print("\n" + "=" * 70)
        print("CO2RR KNOWLEDGE GRAPH — STATISTICS")
        print("=" * 70)

        # Node counts by type
        type_counts = Counter()
        for _, data in self.G.nodes(data=True):
            type_counts[data.get("entity_type", "Unknown")] += 1
        print(f"\n--- Node Types ---")
        for t, c in type_counts.most_common():
            print(f"  {t:20s}: {c:>5d}")
        print(f"  {'TOTAL':20s}: {self.G.number_of_nodes():>5d}")

        # Edge counts by type
        edge_counts = Counter()
        for _, _, data in self.G.edges(data=True):
            edge_counts[data.get("relation_type", "Unknown")] += 1
        print(f"\n--- Edge Types ---")
        for t, c in edge_counts.most_common():
            print(f"  {t:20s}: {c:>5d}")
        print(f"  {'TOTAL':20s}: {self.G.number_of_edges():>5d}")

        # Catalyst type distribution
        cat_types = Counter()
        for _, data in self.G.nodes(data=True):
            if data.get("entity_type") == "Catalyst":
                cat_types[data.get("catalyst_type", "Other")] += 1
        print(f"\n--- Catalyst Type Distribution ---")
        for t, c in cat_types.most_common():
            print(f"  {t:25s}: {c:>4d}")

        # Product distribution
        prod_mentions = Counter()
        for _, v, data in self.G.edges(data=True):
            if data.get("relation_type") == "PRODUCES":
                prod_label = self.G.nodes[v].get("label", v)
                prod_mentions[prod_label] += 1
        print(f"\n--- Product Mentions (top 15) ---")
        for prod, cnt in prod_mentions.most_common(15):
            print(f"  {prod:15s}: {cnt:>4d}")

        # FE coverage
        total_produces = edge_counts.get("PRODUCES", 0)
        with_fe = sum(1 for _, _, d in self.G.edges(data=True)
                      if d.get("relation_type") == "PRODUCES"
                      and "faradaic_efficiency" in d)
        print(f"\n--- Data Completeness ---")
        print(f"  PRODUCES edges with FE:        {with_fe}/{total_produces} "
              f"({100*with_fe/max(total_produces,1):.1f}%)")
        with_pot = sum(1 for _, _, d in self.G.edges(data=True)
                       if d.get("relation_type") == "PRODUCES"
                       and "potential_V" in d)
        print(f"  PRODUCES edges with potential:  {with_pot}/{total_produces} "
              f"({100*with_pot/max(total_produces,1):.1f}%)")

        # Graph connectivity
        print(f"\n--- Graph Properties ---")
        print(f"  Connected components (weak): {nx.number_weakly_connected_components(self.G)}")
        degrees = [d for _, d in self.G.degree()]
        print(f"  Avg degree: {sum(degrees)/len(degrees):.1f}")
        print(f"  Max degree: {max(degrees)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build CO2RR Knowledge Graph")
    parser.add_argument("--query", action="store_true", help="Interactive query mode")
    parser.add_argument("--stats", action="store_true", help="Print KG statistics")
    parser.add_argument("--export-neo4j", action="store_true", help="Export Neo4j script")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build KG
    kg = CO2RRKnowledgeGraph()
    kg.build_from_csv(INPUT_FILE)

    # Always print stats
    kg.print_stats()

    if args.stats:
        return

    # Export all formats
    print("\n" + "=" * 70)
    print("EXPORTING KNOWLEDGE GRAPH")
    print("=" * 70)

    kg.export_graphml(OUTPUT_DIR / "co2rr_kg.graphml")
    kg.export_rdf_turtle(OUTPUT_DIR / "co2rr_kg.ttl")
    kg.export_json_ld(OUTPUT_DIR / "co2rr_kg.jsonld")
    kg.export_neo4j_cypher(OUTPUT_DIR / "neo4j_import.cypher")

    # Save ontology definition
    ontology_path = OUTPUT_DIR / "ontology.json"
    ontology_path.write_text(json.dumps(ONTOLOGY, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Ontology definition → {ontology_path}")

    # Demo queries
    print("\n" + "=" * 70)
    print("DEMO QUERIES")
    print("=" * 70)

    print("\n--- Q1: Top 10 catalysts for CO ---")
    results = kg.query_top_catalysts_for_product("CO", 10)
    if results:
        print(f"{'Catalyst':35s} {'Type':20s} {'FE%':>8s} {'Potential':>15s}")
        print("-" * 80)
        for r in results:
            print(f"{r['catalyst']:35s} {r['catalyst_type']:20s} {r['FE%']:7.1f}% {r['potential']:>15s}")

    print("\n--- Q2: Top 10 catalysts for C2H4 ---")
    results = kg.query_top_catalysts_for_product("C2H4", 10)
    if results:
        print(f"{'Catalyst':35s} {'Type':20s} {'FE%':>8s} {'Potential':>15s}")
        print("-" * 80)
        for r in results:
            print(f"{r['catalyst']:35s} {r['catalyst_type']:20s} {r['FE%']:7.1f}% {r['potential']:>15s}")

    print("\n--- Q3: Top 10 catalysts for HCOOH ---")
    results = kg.query_top_catalysts_for_product("HCOOH", 10)
    if results:
        print(f"{'Catalyst':35s} {'Type':20s} {'FE%':>8s} {'Potential':>15s}")
        print("-" * 80)
        for r in results:
            print(f"{r['catalyst']:35s} {r['catalyst_type']:20s} {r['FE%']:7.1f}% {r['potential']:>15s}")

    print("\n--- Q4: Catalyst type statistics ---")
    type_stats = kg.query_catalyst_type_stats()
    print(f"{'Type':25s} {'#Catalysts':>12s} {'Avg FE%':>10s} {'Top Product':>15s}")
    print("-" * 65)
    for cat_type, stats in sorted(type_stats.items(), key=lambda x: -x[1]["count"]):
        avg_fe = sum(stats["avg_fe"]) / len(stats["avg_fe"]) if stats["avg_fe"] else 0
        top_prod = stats["products"].most_common(1)[0][0] if stats["products"] else "N/A"
        print(f"{cat_type:25s} {stats['count']:>12d} {avg_fe:>9.1f}% {top_prod:>15s}")

    if args.query:
        print("\n" + "=" * 70)
        print("INTERACTIVE QUERY MODE (type 'exit' to quit)")
        print("=" * 70)
        print("Commands:")
        print("  product <name>     — Top catalysts for a product")
        print("  catalyst <name>    — All products of a catalyst")
        print("  stats              — Full statistics")

        while True:
            try:
                cmd = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd.lower() in ("exit", "quit", "q"):
                break
            parts = cmd.split(maxsplit=1)
            if not parts:
                continue
            if parts[0] == "product" and len(parts) > 1:
                results = kg.query_top_catalysts_for_product(parts[1])
                for r in results:
                    print(f"  {r['catalyst']:35s} FE={r['FE%']:.1f}%  {r['potential']}")
            elif parts[0] == "catalyst" and len(parts) > 1:
                results = kg.query_catalyst_products(parts[1])
                for r in results:
                    fe = f"FE={r['FE%']:.1f}%" if r['FE%'] else ""
                    print(f"  → {r['product']:15s} {fe:>12s}  {r['potential']}")
            elif parts[0] == "stats":
                kg.print_stats()
            else:
                print("Unknown command. Try: product CO, catalyst Cu, stats")


if __name__ == "__main__":
    main()
