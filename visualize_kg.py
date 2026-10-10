"""
visualize_kg.py
===============
Visualize the CO2RR knowledge graph with multiple views.

Usage:
    python visualize_kg.py
"""

import html
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

OUTPUT_DIR = Path(__file__).resolve().parent / "relation_extraction_results" / "knowledge_graph"
GRAPHML_PATH = OUTPUT_DIR / "co2rr_kg.graphml"
NORM_CSV = Path(__file__).resolve().parent / "relation_extraction_results" / "normalized_relations.csv"
TITLE_LOOKUP_FILE = Path(__file__).resolve().parent / "titles_dois.csv"


def normalize_paper_title(title: str) -> str:
    text = html.unescape(str(title))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


@lru_cache(maxsize=1)
def load_paper_title_lookup():
    exact_titles = {}
    ordered_titles = []
    if not TITLE_LOOKUP_FILE.exists():
        return exact_titles, tuple(ordered_titles)

    try:
        df = pd.read_csv(TITLE_LOOKUP_FILE, usecols=["query_title"])
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


def resolve_paper_title(title: str) -> str:
    normalized = normalize_paper_title(title)
    if not normalized or normalized.lower() == "nan":
        return ""
    exact_titles, ordered_titles = load_paper_title_lookup()
    resolved = exact_titles.get(normalized)
    if resolved:
        return resolved

    prefix_matches = [full_title for full_title in ordered_titles
                      if full_title.startswith(normalized)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return normalized


def collect_catalyst_papers(G, catalyst_uris):
    cat_papers = defaultdict(set)
    for cat_uri in set(catalyst_uris):
        for _, target, edata in G.out_edges(cat_uri, data=True):
            relation_type = edata.get("relation_type")
            if relation_type == "REPORTED_IN":
                paper_data = G.nodes[target]
                paper_title = resolve_paper_title(
                    paper_data.get("full_title") or paper_data.get("label", "")
                )
                if paper_title:
                    cat_papers[cat_uri].add(paper_title)
                continue

            if relation_type == "PRODUCES":
                paper_title = resolve_paper_title(edata.get("paper", ""))
                if paper_title:
                    cat_papers[cat_uri].add(paper_title)

    return cat_papers


def load_graph():
    G = nx.read_graphml(str(GRAPHML_PATH))
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: Core KG structure — Products as hub, top catalysts connected
# ═══════════════════════════════════════════════════════════════════════════════
def plot_core_kg(G):
    """Interactive knowledge graph using pyvis — open in browser to explore."""
    from pyvis.network import Network

    TYPE_COLORS = {
        "Product": "#E74C3C",
        "MetalCatalyst": "#1ABC9C",
        "AlloyCatalyst": "#3498DB",
        "MetalOxideCatalyst": "#27AE60",
        "CarbonBasedCatalyst": "#F39C12",
        "MOFCatalyst": "#9B59B6",
        "SAC": "#E67E22",
        "CompositeCatalyst": "#2980B9",
        "OtherCatalyst": "#95A5A6",
    }

    # Get product nodes
    products = [n for n, d in G.nodes(data=True) if d.get("entity_type") == "Product"]
    product_labels = {n: G.nodes[n].get("label", n) for n in products}

    # For each product, find top 5 catalysts by FE
    sub_edges = []
    for prod in products:
        preds = list(G.predecessors(prod))
        cat_preds = [(p, G.nodes[p]) for p in preds
                     if G.nodes[p].get("entity_type") == "Catalyst"]
        cat_fe = {}
        for cat_uri, cat_data in cat_preds:
            for _, tgt, edata in G.edges(cat_uri, data=True):
                if tgt == prod:
                    try:
                        fe = float(edata.get("faradaic_efficiency",
                                             edata.get("fe_pct", 0)) or 0)
                    except (ValueError, TypeError):
                        fe = 0
                    if cat_uri not in cat_fe or fe > cat_fe[cat_uri]:
                        cat_fe[cat_uri] = fe
        top_cats = sorted(cat_fe.items(), key=lambda x: -x[1])[:5]
        for cat_uri, fe in top_cats:
            sub_edges.append((cat_uri, prod, fe))

    # Build pyvis network
    net = Network(height="900px", width="100%", directed=True,
                  bgcolor="#FAFAFA", font_color="#333")
    net.barnes_hut(gravity=-3000, central_gravity=0.3,
                   spring_length=200, spring_strength=0.01)
    net.set_options('{"interaction": {"hover": true, "tooltipDelay": 100}}')

    added_nodes = set()
    # Add product nodes (large, red)
    for prod in products:
        label = product_labels[prod]
        net.add_node(prod, label=label, color=TYPE_COLORS["Product"],
                     size=40, font={"size": 18, "face": "Arial", "color": "#fff"},
                     shape="dot", borderWidth=2, borderWidthSelected=4,
                     title=f"{label} (Product)")
        added_nodes.add(prod)

    cat_papers = collect_catalyst_papers(
        G,
        [cat_uri for cat_uri, _, _ in sub_edges],
    )

    # Add catalyst nodes + edges
    for cat_uri, prod_uri, fe in sub_edges:
        if cat_uri not in added_nodes:
            data = G.nodes[cat_uri]
            cat_type = data.get("catalyst_type", "OtherCatalyst")
            cat_label = data.get("label", cat_uri)
            color = TYPE_COLORS.get(cat_type, "#95A5A6")
            type_name = cat_type.replace("Catalyst", "")
            fe_str = f"FE = {fe:.1f}%" if fe > 0 else ""
            papers = cat_papers.get(cat_uri, set())
            paper_str = "\n".join(f"  - {p}" for p in sorted(papers)) if papers else ""
            tooltip_lines = [cat_label, f"Type: {type_name}"]
            if fe_str:
                tooltip_lines.append(fe_str)
            tooltip = "\n".join(tooltip_lines)
            if paper_str:
                tooltip += f"\n\nPaper(s):\n{paper_str}"
            net.add_node(cat_uri, label=cat_label, color=color,
                         size=20, font={"size": 12, "face": "Arial"},
                         shape="dot", borderWidth=1,
                         title=tooltip)
            added_nodes.add(cat_uri)

        # Edge — show FE + paper on hover
        fe_str = f"FE={fe:.0f}%" if fe > 0 else ""
        papers = cat_papers.get(cat_uri, set())
        edge_parts = []
        if fe_str:
            edge_parts.append(fe_str)
        if papers:
            edge_parts.append("Paper(s):\n" + "\n".join(f"  - {p}" for p in sorted(papers)))
        edge_title = "\n\n".join(edge_parts)
        edge_width = max(1, fe / 20)
        net.add_edge(cat_uri, prod_uri, title=edge_title,
                     width=edge_width, color={"color": "#888", "opacity": 0.6},
                     arrows="to", smooth={"type": "curvedCW", "roundness": 0.2})

    # Add legend as invisible nodes positioned at top-left
    legend_items = [
        ("Product", TYPE_COLORS["Product"]),
        ("Alloy", TYPE_COLORS["AlloyCatalyst"]),
        ("Metal Oxide", TYPE_COLORS["MetalOxideCatalyst"]),
        ("Carbon-based", TYPE_COLORS["CarbonBasedCatalyst"]),
        ("Composite", TYPE_COLORS["CompositeCatalyst"]),
        ("MOF", TYPE_COLORS["MOFCatalyst"]),
        ("SAC", TYPE_COLORS["SAC"]),
        ("Pure Metal", TYPE_COLORS["MetalCatalyst"]),
        ("Other", TYPE_COLORS["OtherCatalyst"]),
    ]
    x_leg, y_leg = -1200, -500
    for i, (name, color) in enumerate(legend_items):
        lid = f"_legend_{i}"
        net.add_node(lid, label=f"  {name}", color=color,
                     size=10, shape="dot", x=x_leg, y=y_leg + i * 50,
                     physics=False, font={"size": 14, "align": "left"})

    path = OUTPUT_DIR / "kg_interactive.html"
    net.save_graph(str(path))

    # Post-process: inject CSS so tooltip shows full text with wrapping
    html = path.read_text(encoding="utf-8")
    # Replace vis.js default tooltip with a custom one that supports full-width text
    tooltip_patch = r"""
    <style>
    #custom-tooltip {
        position: fixed;
        display: none;
        background: #f5f4ed;
        border: 1px solid #808074;
        border-radius: 4px;
        padding: 8px 12px;
        max-width: 50vw;
        max-height: 80vh;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
        font-family: Arial, sans-serif;
        font-size: 13px;
        line-height: 1.5;
        pointer-events: none;
        z-index: 9999;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
    }
    div.vis-tooltip { display: none !important; }
    </style>
    <div id="custom-tooltip"></div>
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        var tip = document.getElementById("custom-tooltip");
        var container = document.getElementById("mynetwork");
        if (!container) return;
        // wait for network to be ready
        var waitNet = setInterval(function() {
            if (typeof network === "undefined") return;
            clearInterval(waitNet);
            network.on("hoverNode", function(params) {
                var nodeId = params.node;
                var nodeData = network.body.data.nodes.get(nodeId);
                if (nodeData && nodeData.title) {
                    tip.textContent = nodeData.title;
                    tip.style.display = "block";
                }
            });
            network.on("blurNode", function() {
                tip.style.display = "none";
            });
            network.on("hoverEdge", function(params) {
                var edgeId = params.edge;
                var edgeData = network.body.data.edges.get(edgeId);
                if (edgeData && edgeData.title) {
                    tip.textContent = edgeData.title;
                    tip.style.display = "block";
                }
            });
            network.on("blurEdge", function() {
                tip.style.display = "none";
            });
            container.addEventListener("mousemove", function(e) {
                if (tip.style.display === "block") {
                    var x = e.clientX + 15;
                    var y = e.clientY + 15;
                    var tw = tip.offsetWidth;
                    var th = tip.offsetHeight;
                    var vw = window.innerWidth;
                    var vh = window.innerHeight;
                    if (x + tw > vw - 10) x = e.clientX - tw - 15;
                    if (y + th > vh - 10) y = e.clientY - th - 15;
                    if (x < 5) x = 5;
                    if (y < 5) y = 5;
                    tip.style.left = x + "px";
                    tip.style.top = y + "px";
                }
            });
        }, 200);
    });
    </script>
    """
    html = html.replace("</body>", tooltip_patch + "\n</body>", 1)
    path.write_text(html, encoding="utf-8")
    print(f"Saved → {path}  (open in browser)")

    # Also keep a static PNG via matplotlib for the paper
    _plot_core_kg_static(G)


def _plot_core_kg_static(G):
    """Static matplotlib version for paper figures."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    products = [n for n, d in G.nodes(data=True) if d.get("entity_type") == "Product"]
    product_labels = {n: G.nodes[n].get("label", n) for n in products}

    sub_nodes = set(products)
    sub_edges = []
    for prod in products:
        preds = list(G.predecessors(prod))
        cat_preds = [(p, G.nodes[p]) for p in preds
                     if G.nodes[p].get("entity_type") == "Catalyst"]
        cat_fe = {}
        for cat_uri, cat_data in cat_preds:
            for _, tgt, edata in G.edges(cat_uri, data=True):
                if tgt == prod:
                    try:
                        fe = float(edata.get("faradaic_efficiency",
                                             edata.get("fe_pct", 0)) or 0)
                    except (ValueError, TypeError):
                        fe = 0
                    if cat_uri not in cat_fe or fe > cat_fe[cat_uri]:
                        cat_fe[cat_uri] = fe
        top_cats = sorted(cat_fe.items(), key=lambda x: -x[1])[:5]
        for cat_uri, fe in top_cats:
            sub_nodes.add(cat_uri)
            sub_edges.append((cat_uri, prod, fe))

    SG = nx.DiGraph()
    for n in sub_nodes:
        SG.add_node(n, **dict(G.nodes[n]))
    for u, v, fe in sub_edges:
        SG.add_edge(u, v, fe=fe)

    pos = {}
    n_prods = len(products)
    for i, p in enumerate(sorted(products, key=lambda x: product_labels[x])):
        angle = 2 * np.pi * i / n_prods
        pos[p] = (2.5 * np.cos(angle), 2.5 * np.sin(angle))
    np.random.seed(42)
    for u, v, fe in sub_edges:
        if u not in pos:
            px, py = pos[v]
            angle = np.arctan2(py, px)
            noise = np.random.uniform(-0.3, 0.3)
            r = 5.0 + np.random.uniform(0, 1.5)
            pos[u] = (r * np.cos(angle + noise), r * np.sin(angle + noise))

    TYPE_COLORS = {
        "Product": "#FF6B6B", "MetalCatalyst": "#4ECDC4",
        "AlloyCatalyst": "#45B7D1", "MetalOxideCatalyst": "#96CEB4",
        "CarbonBasedCatalyst": "#FFEAA7", "MOFCatalyst": "#DDA0DD",
        "SAC": "#FF8C00", "CompositeCatalyst": "#87CEEB", "OtherCatalyst": "#D3D3D3",
    }

    node_colors = []
    node_sizes = []
    labels = {}
    for n in SG.nodes():
        data = SG.nodes[n]
        if data.get("entity_type") == "Product":
            node_colors.append(TYPE_COLORS["Product"])
            node_sizes.append(1200)
            labels[n] = data.get("label", n)
        else:
            cat_type = data.get("catalyst_type", "OtherCatalyst")
            node_colors.append(TYPE_COLORS.get(cat_type, "#D3D3D3"))
            node_sizes.append(400)
            labels[n] = data.get("label", n)

    nx.draw_networkx_nodes(SG, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_labels(SG, pos, labels, font_size=6, ax=ax)

    for u, v, edata in SG.edges(data=True):
        # Edge color matches catalyst node color
        cat_type = SG.nodes[u].get("catalyst_type", "OtherCatalyst")
        edge_color = TYPE_COLORS.get(cat_type, "#D3D3D3")
        ax.annotate("", xy=pos[v], xytext=pos[u],
                     arrowprops=dict(arrowstyle="-|>", color=edge_color,
                                     lw=1.0, alpha=0.6,
                                     connectionstyle="arc3,rad=0.15"))

    legend_items = [
        mpatches.Patch(color=TYPE_COLORS["Product"], label="Product"),
        mpatches.Patch(color=TYPE_COLORS["AlloyCatalyst"], label="Alloy"),
        mpatches.Patch(color=TYPE_COLORS["MetalOxideCatalyst"], label="Metal Oxide"),
        mpatches.Patch(color=TYPE_COLORS["CarbonBasedCatalyst"], label="Carbon-based"),
        mpatches.Patch(color=TYPE_COLORS["CompositeCatalyst"], label="Composite"),
        mpatches.Patch(color=TYPE_COLORS["MOFCatalyst"], label="MOF"),
        mpatches.Patch(color=TYPE_COLORS["SAC"], label="SAC"),
        mpatches.Patch(color=TYPE_COLORS["MetalCatalyst"], label="Pure Metal"),
        mpatches.Patch(color=TYPE_COLORS["OtherCatalyst"], label="Other"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("CO₂RR Knowledge Graph — Core Structure\n(Top 5 catalysts per product by FE%)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    path = OUTPUT_DIR / "kg_core_structure.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: Catalyst type → Product heatmap (Sankey-style)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_type_product_heatmap():
    """Heatmap: catalyst type × product → average FE%."""
    df = pd.read_csv(NORM_CSV)
    df = df[df["is_group_label"] == False].copy()
    df = df[df["fe_pct"].notna()].copy()

    # Reload to get catalyst types
    from build_knowledge_graph import classify_catalyst
    df["catalyst_type"] = df["catalyst"].apply(classify_catalyst)

    main_products = ["CO", "HCOOH", "C2H4", "CH4", "C2H5OH", "CH3OH", "H2",
                     "n-propanol", "CH3COOH", "C2H6"]
    df = df[df["product"].isin(main_products)]

    type_order = ["AlloyCatalyst", "MetalOxideCatalyst", "CompositeCatalyst",
                  "CarbonBasedCatalyst", "MOFCatalyst", "SAC", "MetalCatalyst",
                  "OtherCatalyst"]

    # Build matrix: avg FE
    matrix = np.full((len(type_order), len(main_products)), np.nan)
    count_matrix = np.zeros((len(type_order), len(main_products)), dtype=int)

    for i, ct in enumerate(type_order):
        for j, prod in enumerate(main_products):
            subset = df[(df["catalyst_type"] == ct) & (df["product"] == prod)]
            if len(subset) > 0:
                matrix[i, j] = subset["fe_pct"].mean()
                count_matrix[i, j] = len(subset)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    # Labels
    type_labels = [t.replace("Catalyst", "").replace("Metal", "Metal ") for t in type_order]
    ax.set_xticks(range(len(main_products)))
    ax.set_xticklabels(main_products, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(type_order)))
    ax.set_yticklabels(type_labels, fontsize=10)

    # Annotate cells
    for i in range(len(type_order)):
        for j in range(len(main_products)):
            if not np.isnan(matrix[i, j]):
                color = "white" if matrix[i, j] > 60 else "black"
                ax.text(j, i, f"{matrix[i, j]:.0f}%\n(n={count_matrix[i, j]})",
                        ha="center", va="center", fontsize=7, color=color,
                        fontweight="bold")

    plt.colorbar(im, ax=ax, label="Average Faradaic Efficiency (%)", shrink=0.8)
    ax.set_title("Catalyst Type × Product — Average Faradaic Efficiency",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("CO₂RR Product", fontsize=11)
    ax.set_ylabel("Catalyst Type", fontsize=11)

    path = OUTPUT_DIR / "kg_type_product_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: Ontology schema diagram
# ═══════════════════════════════════════════════════════════════════════════════
def plot_ontology():
    """Visualize the CO2RR ontology schema."""
    fig, ax = plt.subplots(figsize=(14, 10))

    G = nx.DiGraph()

    # Class hierarchy
    G.add_node("Catalyst", level=0)
    for sub in ["Metal", "Alloy", "MetalOxide", "CarbonBased", "MOF", "SAC", "Composite"]:
        G.add_node(sub, level=1)
        G.add_edge(sub, "Catalyst", relation="IS_A")

    G.add_node("Product", level=0)
    for sub in ["C1Product", "C2Product", "C3+Product", "HER (H₂)"]:
        G.add_node(sub, level=1)
        G.add_edge(sub, "Product", relation="IS_A")

    G.add_node("Paper", level=0)
    G.add_node("Electrolyte", level=0)

    # Relations
    G.add_edge("Catalyst", "Product", relation="PRODUCES")
    G.add_edge("Catalyst", "Paper", relation="REPORTED_IN")
    G.add_edge("Catalyst", "Electrolyte", relation="USES_ELECTROLYTE")

    # Manual layout
    pos = {
        # Top-level classes
        "Catalyst": (-2, 3),
        "Product": (5, 3),
        "Paper": (-2, -2),
        "Electrolyte": (5, -2),
        # Catalyst subtypes
        "Metal": (-5.5, 0.5),
        "Alloy": (-4, 0.5),
        "MetalOxide": (-2.5, 0.5),
        "CarbonBased": (-1, 0.5),
        "MOF": (0.3, 0.5),
        "SAC": (-4.7, -0.5),
        "Composite": (-3.2, -0.5),
        # Product subtypes
        "C1Product": (3.5, 0.5),
        "C2Product": (5, 0.5),
        "C3+Product": (6.5, 0.5),
        "HER (H₂)": (5, -0.3),
    }

    # Draw IS_A edges (dashed)
    isa_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "IS_A"]
    nx.draw_networkx_edges(G, pos, edgelist=isa_edges, style="dashed",
                           edge_color="#aaa", arrows=True,
                           arrowstyle="-|>", ax=ax, connectionstyle="arc3,rad=0.05")

    # Draw relation edges (solid, colored)
    rel_colors = {"PRODUCES": "#E74C3C", "REPORTED_IN": "#3498DB", "USES_ELECTROLYTE": "#2ECC71"}
    for u, v, d in G.edges(data=True):
        rel = d.get("relation", "")
        if rel in rel_colors:
            ax.annotate("", xy=pos[v], xytext=pos[u],
                         arrowprops=dict(arrowstyle="-|>", color=rel_colors[rel],
                                         lw=2.5, connectionstyle="arc3,rad=0.15"))
            mx = (pos[u][0] + pos[v][0]) / 2
            my = (pos[u][1] + pos[v][1]) / 2 + 0.3
            ax.text(mx, my, rel, fontsize=8, ha="center", color=rel_colors[rel],
                    fontweight="bold", fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=rel_colors[rel], alpha=0.9))

    # Node colors
    class_colors = {
        "Catalyst": "#3498DB", "Product": "#E74C3C",
        "Paper": "#9B59B6", "Electrolyte": "#2ECC71",
    }
    sub_color = "#ECF0F1"

    for node in G.nodes():
        x, y = pos[node]
        if node in class_colors:
            color = class_colors[node]
            bbox = dict(boxstyle="round,pad=0.5", facecolor=color,
                        edgecolor="black", alpha=0.85, lw=2)
            ax.text(x, y, node, fontsize=12, ha="center", va="center",
                    fontweight="bold", color="white", bbox=bbox)
        else:
            bbox = dict(boxstyle="round,pad=0.3", facecolor=sub_color,
                        edgecolor="#888", alpha=0.9)
            ax.text(x, y, node, fontsize=9, ha="center", va="center",
                    color="#333", bbox=bbox)

    # Edge properties annotation
    ax.text(1.5, 4.2, "Edge Properties:", fontsize=9, fontweight="bold")
    ax.text(1.5, 3.8, "PRODUCES: faradaic_efficiency, potential, current_density", fontsize=8)
    ax.text(1.5, 3.5, "REPORTED_IN: (links catalyst to paper)", fontsize=8)
    ax.text(1.5, 3.2, "USES_ELECTROLYTE: (links catalyst to electrolyte)", fontsize=8)

    # IS_A legend
    ax.annotate("", xy=(-1, -2.7), xytext=(-2, -2.7),
                arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=1, linestyle="dashed"))
    ax.text(-0.8, -2.7, "IS_A (subclass)", fontsize=8, va="center", color="#888")

    ax.set_xlim(-7, 8.5)
    ax.set_ylim(-3.5, 5)
    ax.set_title("CO₂RR Knowledge Graph — Ontology Schema", fontsize=14, fontweight="bold")
    ax.axis("off")

    path = OUTPUT_DIR / "kg_ontology_schema.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: FE distribution violin plot by catalyst type
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fe_by_catalyst_type():
    """Violin plot: FE distribution by catalyst type."""
    df = pd.read_csv(NORM_CSV)
    df = df[(df["is_group_label"] == False) & df["fe_pct"].notna()].copy()

    from build_knowledge_graph import classify_catalyst
    df["catalyst_type"] = df["catalyst"].apply(classify_catalyst)

    type_order = ["SAC", "AlloyCatalyst", "CarbonBasedCatalyst", "CompositeCatalyst",
                  "MetalOxideCatalyst", "MOFCatalyst", "OtherCatalyst", "MetalCatalyst"]
    type_labels = ["SAC", "Alloy", "Carbon", "Composite", "Metal Oxide", "MOF", "Other", "Metal"]

    data_groups = []
    valid_labels = []
    for ct, lab in zip(type_order, type_labels):
        vals = df[df["catalyst_type"] == ct]["fe_pct"].values
        if len(vals) >= 3:
            data_groups.append(vals)
            valid_labels.append(f"{lab}\n(n={len(vals)})")

    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(data_groups, showmeans=True, showmedians=True)

    colors = ["#FF8C00", "#45B7D1", "#FFEAA7", "#87CEEB", "#96CEB4", "#DDA0DD", "#D3D3D3", "#4ECDC4"]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("red")
    parts["cmedians"].set_color("black")

    ax.set_xticks(range(1, len(valid_labels) + 1))
    ax.set_xticklabels(valid_labels, fontsize=9)
    ax.set_ylabel("Faradaic Efficiency (%)", fontsize=11)
    ax.set_title("FE Distribution by Catalyst Type", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="#ccc", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Add mean labels
    for i, vals in enumerate(data_groups):
        ax.text(i + 1, np.mean(vals) + 2, f"{np.mean(vals):.0f}%",
                ha="center", fontsize=8, color="red", fontweight="bold")

    path = OUTPUT_DIR / "kg_fe_by_catalyst_type.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: KG summary dashboard
# ═══════════════════════════════════════════════════════════════════════════════
def plot_kg_summary():
    """Summary dashboard with multiple panels."""
    df = pd.read_csv(NORM_CSV)
    df = df[df["is_group_label"] == False].copy()

    from build_knowledge_graph import classify_catalyst
    df["catalyst_type"] = df["catalyst"].apply(classify_catalyst)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Catalyst type distribution (pie)
    ax = axes[0, 0]
    type_counts = df["catalyst_type"].value_counts()
    type_labels_map = {
        "AlloyCatalyst": "Alloy", "MetalOxideCatalyst": "Metal Oxide",
        "CompositeCatalyst": "Composite", "CarbonBasedCatalyst": "Carbon",
        "MOFCatalyst": "MOF", "SAC": "SAC", "MetalCatalyst": "Metal",
        "OtherCatalyst": "Other",
    }
    labels = [type_labels_map.get(t, t) for t in type_counts.index]
    colors = ["#45B7D1", "#96CEB4", "#87CEEB", "#FFEAA7", "#DDA0DD",
              "#FF8C00", "#4ECDC4", "#D3D3D3"]
    wedges, texts, autotexts = ax.pie(type_counts.values, labels=labels,
                                       autopct="%1.0f%%", colors=colors[:len(labels)],
                                       textprops={"fontsize": 8})
    ax.set_title("Catalyst Type Distribution", fontsize=11, fontweight="bold")

    # Panel 2: Product frequency (bar)
    ax = axes[0, 1]
    prod_counts = df["product"].value_counts().head(10)
    bars = ax.barh(range(len(prod_counts)), prod_counts.values, color="#E74C3C", alpha=0.8)
    ax.set_yticks(range(len(prod_counts)))
    ax.set_yticklabels(prod_counts.index, fontsize=9)
    ax.set_xlabel("Number of Relations", fontsize=9)
    ax.set_title("Product Frequency (top 10)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, prod_counts.values):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2, str(val),
                va="center", fontsize=8)

    # Panel 3: FE availability by product (stacked bar)
    ax = axes[1, 0]
    main_prods = ["CO", "HCOOH", "C2H4", "CH4", "C2H5OH", "CH3OH", "H2"]
    with_fe = []
    without_fe = []
    for p in main_prods:
        subset = df[df["product"] == p]
        n_with = subset["fe_pct"].notna().sum()
        n_without = len(subset) - n_with
        with_fe.append(n_with)
        without_fe.append(n_without)

    x = range(len(main_prods))
    ax.bar(x, with_fe, color="#2ECC71", label="With FE", alpha=0.8)
    ax.bar(x, without_fe, bottom=with_fe, color="#E74C3C", label="Without FE", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(main_prods, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Data Completeness by Product", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 4: KG summary stats text
    ax = axes[1, 1]
    ax.axis("off")
    G = load_graph()
    stats_text = (
        f"CO₂RR Knowledge Graph Summary\n"
        f"{'─' * 35}\n"
        f"Total Nodes:          {G.number_of_nodes():,}\n"
        f"  • Catalysts:        601\n"
        f"  • Papers:           566\n"
        f"  • Electrolytes:     79\n"
        f"  • Products:         25\n"
        f"{'─' * 35}\n"
        f"Total Edges:          {G.number_of_edges():,}\n"
        f"  • PRODUCES:         950\n"
        f"  • REPORTED_IN:      950\n"
        f"  • USES_ELECTROLYTE: 950\n"
        f"{'─' * 35}\n"
        f"Ontology Classes:     4 (+15 sub)\n"
        f"Relation Types:       3 (+IS_A)\n"
        f"FE Coverage:          69.7%\n"
        f"Potential Coverage:   54.8%\n"
        f"{'─' * 35}\n"
        f"Export Formats:\n"
        f"  GraphML / RDF Turtle / JSON-LD\n"
        f"  Neo4j Cypher import script"
    )
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    fig.suptitle("CO₂RR Knowledge Graph — Overview Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUTPUT_DIR / "kg_dashboard.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def main():
    print("=" * 60)
    print("CO2RR Knowledge Graph Visualization")
    print("=" * 60)

    plot_ontology()
    plot_core_kg(load_graph())
    plot_type_product_heatmap()
    plot_fe_by_catalyst_type()
    plot_kg_summary()

    print("\nAll visualizations saved to:")
    print(f"  {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
