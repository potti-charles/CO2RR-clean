"""
extract_relations.py
====================
Zero-shot relation extraction from CO2RR literature using Qwen2.5-7B-Instruct.
Extracts structured triplets: (catalyst, produces, product, FE%, potential, electrolyte).

This goes beyond entity extraction (Paper 1) by capturing the *relationships*
between entities — which catalyst produces which product at what efficiency.

Usage:
    # Extract from training data (title+abstract):
    python extract_relations.py --source data --max_samples 20

    # Extract from PDFs:
    python extract_relations.py --source pdf --pdf_dir gold_corpus/pdfs --max_papers 10

    # Evaluate against gold corpus:
    python extract_relations.py --evaluate

    # Build knowledge graph visualization:
    python extract_relations.py --build_kg
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model_path_utils import (
    QWEN_CACHE_DIR_NAME,
    QWEN_MODEL_ID,
    default_hf_cache_root,
    resolve_hf_model_path,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "relation_extraction_results"
DATA_FILE = REPO_ROOT / "data" / "instruction_tuning_data_json_clean.json"
GOLD_XLS = REPO_ROOT / "gold_corpus" / "gold_corpus.xls"

SYSTEM_PROMPT = (
    "You are a CO2 electroreduction relation extraction assistant. "
    "Extract structured catalyst-product-performance triplets from scientific text. "
    "Return only strict JSON. Do not output any extra text."
)

RELATION_SCHEMA = """\
Extract ALL catalyst-product-performance relationships from the text below.

For each relationship, capture:
- catalyst: the specific catalyst material (e.g. "Cu2O", "Ag-Cu bimetallic", "N-doped carbon")
- product: the CO2 reduction product (e.g. "CO", "HCOOH", "C2H4", "formate", "CH4")
- faradaic_efficiency: the FE value as a string (e.g. "92.3%"), or null if not reported
- potential: the applied potential (e.g. "-0.65 V vs RHE"), or null if not reported
- current_density: current density if mentioned (e.g. "15.2 mA/cm2"), or null
- electrolyte: electrolyte solution if mentioned (e.g. "0.1 M KHCO3"), or null
- selectivity_note: any selectivity or comparison note (e.g. "main product", "suppressed HER"), or null

Return a JSON object:
{
  "relations": [
    {
      "catalyst": "...",
      "product": "...",
      "faradaic_efficiency": "...",
      "potential": "...",
      "current_density": "...",
      "electrolyte": "...",
      "selectivity_note": "..."
    }
  ],
  "paper_summary": "one-sentence summary of the paper's main finding"
}

Rules:
- Only include relationships explicitly stated in the text
- If multiple products are reported for the same catalyst, create separate entries
- Do not guess or infer values not mentioned in the text
- Use chemical formulas where possible (CO not carbon monoxide, HCOOH not formic acid)
- Include the reference electrode in potential (e.g. "vs RHE", "vs Ag/AgCl")
"""


# ---------------------------------------------------------------------------
# Model loading (reuse pattern from verify_silver_with_llm.py)
# ---------------------------------------------------------------------------
def load_model(load_in_4bit: bool = True):
    """Load Qwen2.5-7B-Instruct with optional 4-bit quantization."""
    cache_dir = str(default_hf_cache_root())
    resolved = resolve_hf_model_path(
        QWEN_CACHE_DIR_NAME, fallback_model_id=QWEN_MODEL_ID, cache_dir=cache_dir
    )
    print(f"Loading model: {resolved}")

    tokenizer = AutoTokenizer.from_pretrained(resolved, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(
        resolved, cache_dir=cache_dir, **model_kwargs
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(
    model, tokenizer, user_prompt: str, max_new_tokens: int = 512
) -> str:
    """Generate text using chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def parse_json_response(raw: str) -> Optional[Dict]:
    """Extract JSON object from LLM response, handling truncation."""
    raw = raw.strip()
    # Remove markdown code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    left = raw.find("{")
    right = raw.rfind("}")
    if left == -1 or right == -1 or right <= left:
        return None
    candidate = raw[left : right + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try to fix common issues: trailing comma
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# Relation extraction from text
# ---------------------------------------------------------------------------
def extract_relations_from_text(
    model, tokenizer, text: str, max_new_tokens: int = 512
) -> Dict:
    """Extract relation triplets from a single paper's title+abstract."""
    prompt = f"{RELATION_SCHEMA}\n\nText:\n{text}"
    raw = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    parsed = parse_json_response(raw)

    if parsed is None:
        return {
            "success": False,
            "relations": [],
            "paper_summary": "",
            "raw_output": raw[:500],
        }

    relations = parsed.get("relations", [])
    # Validate and clean relations
    clean_relations = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        catalyst = rel.get("catalyst", "")
        product = rel.get("product", "")
        if not catalyst or not product:
            continue
        clean_relations.append(
            {
                "catalyst": str(catalyst).strip(),
                "product": str(product).strip(),
                "faradaic_efficiency": rel.get("faradaic_efficiency"),
                "potential": rel.get("potential"),
                "current_density": rel.get("current_density"),
                "electrolyte": rel.get("electrolyte"),
                "selectivity_note": rel.get("selectivity_note"),
            }
        )

    return {
        "success": True,
        "relations": clean_relations,
        "paper_summary": parsed.get("paper_summary", ""),
        "raw_output": raw[:500],
    }


# ---------------------------------------------------------------------------
# Data source: instruction tuning data (title+abstract)
# ---------------------------------------------------------------------------
def load_instruction_data(max_samples: int = -1) -> List[Dict]:
    """Load title+abstract from instruction tuning data."""
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if max_samples > 0:
        data = data[:max_samples]
    return data


# ---------------------------------------------------------------------------
# Data source: PDFs
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    import fitz

    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Knowledge graph construction
# ---------------------------------------------------------------------------
def build_knowledge_graph(results: List[Dict]) -> Dict:
    """Build a knowledge graph from extraction results."""
    nodes = {}  # id -> {type, label, properties}
    edges = []  # {source, target, relation, properties}

    catalyst_counter = Counter()
    product_counter = Counter()
    relation_counter = Counter()

    for paper in results:
        for rel in paper.get("relations", []):
            cat = rel["catalyst"]
            prod = rel["product"]
            cat_key = cat.lower().strip()
            prod_key = prod.lower().strip()

            # Add catalyst node
            if cat_key not in nodes:
                nodes[cat_key] = {
                    "id": cat_key,
                    "type": "catalyst",
                    "label": cat,
                    "papers": [],
                }
            nodes[cat_key]["papers"].append(paper.get("source", ""))
            catalyst_counter[cat_key] += 1

            # Add product node
            if prod_key not in nodes:
                nodes[prod_key] = {
                    "id": prod_key,
                    "type": "product",
                    "label": prod,
                    "papers": [],
                }
            nodes[prod_key]["papers"].append(paper.get("source", ""))
            product_counter[prod_key] += 1

            # Add edge
            edge = {
                "source": cat_key,
                "target": prod_key,
                "relation": "produces",
                "faradaic_efficiency": rel.get("faradaic_efficiency"),
                "potential": rel.get("potential"),
                "current_density": rel.get("current_density"),
                "electrolyte": rel.get("electrolyte"),
                "paper": paper.get("source", ""),
            }
            edges.append(edge)
            relation_counter[(cat_key, prod_key)] += 1

    # Deduplicate paper lists
    for node in nodes.values():
        node["papers"] = list(set(node["papers"]))
        node["mention_count"] = len(node["papers"])

    kg = {
        "nodes": list(nodes.values()),
        "edges": edges,
        "statistics": {
            "total_nodes": len(nodes),
            "catalyst_nodes": sum(
                1 for n in nodes.values() if n["type"] == "catalyst"
            ),
            "product_nodes": sum(
                1 for n in nodes.values() if n["type"] == "product"
            ),
            "total_edges": len(edges),
            "unique_relations": len(relation_counter),
            "top_catalysts": catalyst_counter.most_common(20),
            "top_products": product_counter.most_common(20),
            "top_relations": relation_counter.most_common(20),
        },
    }
    return kg


def export_kg_to_csv(kg: Dict, output_dir: Path):
    """Export knowledge graph to CSV files for easy analysis."""
    import pandas as pd

    # Nodes CSV
    nodes_df = pd.DataFrame(kg["nodes"])
    nodes_df.to_csv(output_dir / "kg_nodes.csv", index=False)

    # Edges CSV
    edges_df = pd.DataFrame(kg["edges"])
    edges_df.to_csv(output_dir / "kg_edges.csv", index=False)

    # Summary statistics
    stats = kg["statistics"]
    print(f"\n=== Knowledge Graph Statistics ===")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"  Catalysts: {stats['catalyst_nodes']}")
    print(f"  Products:  {stats['product_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Unique catalyst-product pairs: {stats['unique_relations']}")
    print(f"\nTop 10 catalysts:")
    for cat, cnt in stats["top_catalysts"][:10]:
        print(f"  {cat}: {cnt} mentions")
    print(f"\nTop 10 products:")
    for prod, cnt in stats["top_products"][:10]:
        print(f"  {prod}: {cnt} mentions")
    print(f"\nTop 10 catalyst→product relations:")
    for (cat, prod), cnt in stats["top_relations"][:10]:
        print(f"  {cat} → {prod}: {cnt} papers")


def export_kg_to_networkx(kg: Dict, output_dir: Path):
    """Export knowledge graph to GraphML for visualization in Gephi/Cytoscape."""
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed. Skipping GraphML export.")
        print("Install with: pip install networkx")
        return

    G = nx.DiGraph()

    for node in kg["nodes"]:
        G.add_node(
            node["id"],
            label=node["label"],
            node_type=node["type"],
            mention_count=node.get("mention_count", 1),
        )

    for edge in kg["edges"]:
        fe = edge.get("faradaic_efficiency") or ""
        pot = edge.get("potential") or ""
        edge_label = f"FE={fe}" if fe else "produces"
        G.add_edge(
            edge["source"],
            edge["target"],
            relation=edge["relation"],
            faradaic_efficiency=fe,
            potential=pot,
            current_density=edge.get("current_density") or "",
            label=edge_label,
            paper=edge.get("paper", ""),
        )

    graphml_path = output_dir / "co2rr_knowledge_graph.graphml"
    nx.write_graphml(G, str(graphml_path))
    print(f"\nGraphML exported → {graphml_path}")
    print("Open in Gephi or Cytoscape for interactive visualization.")


# ---------------------------------------------------------------------------
# Evaluation against gold corpus
# ---------------------------------------------------------------------------
def evaluate_against_gold(results: List[Dict], output_dir: Path):
    """Compare extracted relations against gold corpus annotations."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed.")
        return

    if not GOLD_XLS.exists():
        print(f"Gold corpus not found: {GOLD_XLS}")
        return

    gold = pd.read_excel(GOLD_XLS)
    print(f"Gold corpus: {len(gold)} records, {gold['title'].nunique()} papers")

    # Build gold triplets: (title, material, product, efficiency)
    gold_triplets = defaultdict(list)
    for _, row in gold.iterrows():
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        # Depending on gold corpus structure, adapt column names
        material = str(row.get("material", row.get("Materials", ""))).strip()
        product = str(row.get("product", row.get("Products", ""))).strip()
        fe = str(row.get("efficiency", row.get("FE", ""))).strip()
        if material and product:
            gold_triplets[title.lower()].append(
                {"material": material, "product": product, "efficiency": fe}
            )

    # Match extracted results against gold
    matched = 0
    total_extracted = 0
    total_gold = sum(len(v) for v in gold_triplets.values())

    for paper in results:
        source = paper.get("source", "").lower()
        # Try to find matching title in gold
        gold_rels = None
        for gold_title, g_rels in gold_triplets.items():
            if gold_title in source or source in gold_title:
                gold_rels = g_rels
                break

        if gold_rels is None:
            total_extracted += len(paper.get("relations", []))
            continue

        for rel in paper.get("relations", []):
            total_extracted += 1
            cat = rel["catalyst"].lower()
            prod = rel["product"].lower()
            for g_rel in gold_rels:
                g_mat = g_rel["material"].lower()
                g_prod = g_rel["product"].lower()
                # Fuzzy match: check if catalyst/product overlap
                if (cat in g_mat or g_mat in cat) and (
                    prod in g_prod or g_prod in prod
                ):
                    matched += 1
                    break

    report = {
        "total_gold_triplets": total_gold,
        "total_extracted_relations": total_extracted,
        "matched": matched,
        "precision": matched / max(total_extracted, 1),
        "recall": matched / max(total_gold, 1),
    }
    if report["precision"] + report["recall"] > 0:
        report["f1"] = (
            2 * report["precision"] * report["recall"]
            / (report["precision"] + report["recall"])
        )
    else:
        report["f1"] = 0.0

    print(f"\n=== Evaluation Against Gold Corpus ===")
    print(f"Gold triplets: {total_gold}")
    print(f"Extracted relations: {total_extracted}")
    print(f"Matched: {matched}")
    print(f"Precision: {report['precision']:.3f}")
    print(f"Recall: {report['recall']:.3f}")
    print(f"F1: {report['f1']:.3f}")

    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="CO2RR relation extraction and knowledge graph construction."
    )
    parser.add_argument(
        "--source",
        choices=["data", "pdf"],
        default="data",
        help="Input source: 'data' for instruction tuning data, 'pdf' for PDF files.",
    )
    parser.add_argument("--pdf_dir", default="gold_corpus/pdfs")
    parser.add_argument(
        "--max_samples", type=int, default=20, help="Max papers to process (-1 for all)."
    )
    parser.add_argument("--max_papers", type=int, default=-1, help="Alias for max_samples when using PDFs.")
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate against gold corpus.")
    parser.add_argument("--build_kg", action="store_true", help="Build knowledge graph from existing results.")
    parser.add_argument(
        "--results_file",
        default=None,
        help="Path to existing results JSONL for --evaluate or --build_kg.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "relation_results.jsonl"

    # --build_kg mode: load existing results and build KG
    if args.build_kg:
        if args.results_file:
            results_path = Path(args.results_file)
        if not results_path.exists():
            print(f"Results file not found: {results_path}")
            return
        results = []
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        print(f"Loaded {len(results)} papers from {results_path}")
        kg = build_knowledge_graph(results)
        export_kg_to_csv(kg, OUTPUT_DIR)
        export_kg_to_networkx(kg, OUTPUT_DIR)
        with open(OUTPUT_DIR / "knowledge_graph.json", "w", encoding="utf-8") as f:
            json.dump(kg, f, ensure_ascii=False, indent=2)
        print(f"\nKnowledge graph saved → {OUTPUT_DIR / 'knowledge_graph.json'}")
        return

    # --evaluate mode
    if args.evaluate:
        if args.results_file:
            results_path = Path(args.results_file)
        if not results_path.exists():
            print(f"Results file not found: {results_path}")
            return
        results = []
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        evaluate_against_gold(results, OUTPUT_DIR)
        return

    # --- Extraction mode ---
    print("Loading model...")
    model, tokenizer = load_model(load_in_4bit=args.load_in_4bit)

    # Resumable: load already processed sources
    processed_sources = set()
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    processed_sources.add(entry.get("source", ""))
        print(f"Resuming: {len(processed_sources)} papers already processed.")

    if args.source == "data":
        data = load_instruction_data(max_samples=args.max_samples)
        print(f"Processing {len(data)} papers from instruction data...")

        for idx, item in enumerate(data):
            text = item["input"]
            # Preserve the full paper title so downstream graph/tooltips stay intact.
            title_match = re.search(r"Title:(.*?)(?:\n|Abstract:)", text)
            source = title_match.group(1).strip() if title_match else f"sample_{idx}"

            if source in processed_sources:
                continue

            print(f"[{idx+1}/{len(data)}] {source[:60]}...")
            t0 = time.time()
            result = extract_relations_from_text(
                model, tokenizer, text, max_new_tokens=args.max_new_tokens
            )
            elapsed = time.time() - t0

            entry = {
                "source": source,
                "num_relations": len(result["relations"]),
                "relations": result["relations"],
                "paper_summary": result["paper_summary"],
                "success": result["success"],
                "time_sec": round(elapsed, 1),
            }

            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            n = len(result["relations"])
            print(f"  → {n} relation(s) extracted in {elapsed:.1f}s")
            if result["relations"]:
                for r in result["relations"][:3]:
                    fe = r.get("faradaic_efficiency") or "N/A"
                    pot = r.get("potential") or "N/A"
                    print(f"    {r['catalyst']} → {r['product']} (FE={fe}, {pot})")
                if n > 3:
                    print(f"    ... and {n-3} more")

    elif args.source == "pdf":
        pdf_dir = Path(args.pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        max_papers = args.max_papers if args.max_papers > 0 else len(pdf_files)
        pdf_files = pdf_files[:max_papers]

        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return

        print(f"Processing {len(pdf_files)} PDFs...")
        for idx, pdf_path in enumerate(pdf_files):
            source = pdf_path.stem
            if source in processed_sources:
                continue

            print(f"[{idx+1}/{len(pdf_files)}] {pdf_path.name}...")
            try:
                text = extract_text_from_pdf(str(pdf_path))
            except Exception as e:
                print(f"  Error reading PDF: {e}")
                continue

            # Use first ~2000 words (title + abstract + intro + results)
            words = text.split()
            if len(words) > 2000:
                text = " ".join(words[:2000])

            t0 = time.time()
            result = extract_relations_from_text(
                model, tokenizer, text, max_new_tokens=args.max_new_tokens
            )
            elapsed = time.time() - t0

            entry = {
                "source": source,
                "num_relations": len(result["relations"]),
                "relations": result["relations"],
                "paper_summary": result["paper_summary"],
                "success": result["success"],
                "time_sec": round(elapsed, 1),
            }

            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            n = len(result["relations"])
            print(f"  → {n} relation(s) extracted in {elapsed:.1f}s")

    # After extraction, auto-build KG
    print("\n--- Building Knowledge Graph ---")
    results = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    kg = build_knowledge_graph(results)
    export_kg_to_csv(kg, OUTPUT_DIR)
    export_kg_to_networkx(kg, OUTPUT_DIR)
    with open(OUTPUT_DIR / "knowledge_graph.json", "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
