"""Gradio Web UI for CO2RR Catalyst Knowledge Base.

Usage:
    python app.py
    python app.py --port 7860 --share
"""

import argparse
import csv
import hmac
import html
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'gradio'. Install app dependencies with "
        "'python -m pip install -r requirements-app.txt'."
    ) from exc

from co2rr_knowledge_query import (
    answer_single_paper_stream,
    answer_query,
    answer_query_stream,
    build_tfidf_index,
    detect_query_intent,
    get_client,
    load_figure_index,
    load_records,
    search_papers,
    retrieve_figures,
    retrieve_literature,
    retrieve_records,
    DEFAULT_MODEL,
    load_knowledge_base,
)
from runtime_config import OUTPUT_DIR, PROJECT_ROOT
from catalyst_design_agent import run_catalyst_design
from experiment_feedback import (
    create_design_session,
    recent_design_sessions,
    submit_experiment_feedback,
)

# ---------------------------------------------------------------------------
# Pre-load data
# ---------------------------------------------------------------------------

# Credentials are supplied by the student's environment.
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

print("Loading knowledge base and building index...")
build_tfidf_index()
df = load_records()
client = get_client() if DEEPSEEK_API_KEY.strip() else None
print("Ready.")


# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------

def add_user_message(query: str, history: list) -> tuple:
    """Step 1: Immediately show user message in chatbot, clear input."""
    if not query.strip():
        return history, "", ""
    history = history + [{"role": "user", "content": query}]
    return history, "", ""


def _extract_text(content):
    """Extract plain text from Gradio chatbot content (str or list-of-dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Gradio 6 messages format: [{"text": "...", "type": "text"}, ...]
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content) if content else ""


def generate_answer(history: list):
    """Step 2: Stream LLM answer for the last user message. Yields (history, debug_info)."""
    if not history:
        yield history, ""
        return

    # Extract query from last message (handle Gradio 6 content formats)
    last = history[-1]
    if isinstance(last, dict):
        query = _extract_text(last.get("content", ""))
    elif isinstance(last, (list, tuple)):
        query = _extract_text(last[0]) if last[0] else ""
    else:
        query = str(last)

    if not query.strip():
        yield history, ""
        return

    t0 = time.time()

    # Extract prior conversation turns as (user_q, assistant_a) pairs
    chat_pairs = []
    i = 0
    while i < len(history) - 1:  # exclude the last user msg (current query)
        msg_i = history[i]
        msg_next = history[i + 1] if i + 1 < len(history) else None
        role_i = msg_i.get("role", "") if isinstance(msg_i, dict) else ""
        role_next = msg_next.get("role", "") if isinstance(msg_next, dict) else ""
        if role_i == "user" and role_next == "assistant":
            user_text = _extract_text(msg_i.get("content", ""))
            asst_text = _extract_text(msg_next.get("content", ""))
            if user_text and asst_text:
                chat_pairs.append((user_text, asst_text))
            i += 2
        else:
            i += 1

    try:
        # Stream answer chunks with conversation history
        for debug_info, partial_answer in answer_query_stream(
            query, client=client, model=DEFAULT_MODEL, chat_history=chat_pairs
        ):
            updated = history + [{"role": "assistant", "content": partial_answer}]
            yield updated, debug_info

        # Final: add timing info
        elapsed = time.time() - t0
        final_debug = debug_info + f"\n\n耗时: {elapsed:.1f}s"
        yield updated, final_debug

    except Exception as e:
        answer = f"API 调用失败: {e}"
        updated = history + [{"role": "assistant", "content": answer}]
        yield updated, str(e)


def handle_query(query: str, history: list) -> tuple:
    """Legacy single-step handler (kept for compatibility)."""
    history, _, _ = add_user_message(query, history)
    # Consume the streaming generator to get final result
    result = None
    for result in generate_answer(history):
        pass
    if result:
        return result[0], result[1], ""
    return history, "", ""


def generate_and_save_catalyst_design(
    objective: str,
    product: str,
    required: str,
    excluded: str,
    cell: str,
    min_fe: float,
    min_current: float,
    synthesis: str,
):
    """Generate a design and persist the exact task context for feedback."""
    design, evidence = run_catalyst_design(
        objective,
        product,
        required,
        excluded,
        cell,
        min_fe,
        min_current,
        synthesis,
        client=client,
        model=DEFAULT_MODEL,
    )
    if not product or product == "请选择":
        return design, evidence, ""
    constraints = {
        "objective": objective,
        "target_product": product,
        "required_elements": required,
        "excluded_elements": excluded,
        "cell_type": cell,
        "min_fe": float(min_fe or 0),
        "min_current_density": float(min_current or 0),
        "synthesis_preference": synthesis,
    }
    design_id = create_design_session(constraints, design, evidence)
    saved_notice = (
        f"## 设计任务已保存：`{design_id}`\n\n"
        "实验完成后，请在本页下方使用该编号回填结果。\n\n"
    )
    return saved_notice + design, evidence, design_id


def analyze_experiment_feedback(*values):
    """Bind the shared API client to the persistent feedback workflow."""
    return submit_experiment_feedback(
        *values,
        client=client,
        model=DEFAULT_MODEL,
    )


# ---------------------------------------------------------------------------
# Single-paper Q&A handlers
# ---------------------------------------------------------------------------

def select_single_paper(search_text: str):
    """Lock the best DOI/title match and show other candidates for verification."""
    matches = search_papers(search_text, top_k=5)
    if not matches:
        return None, "未找到匹配论文。请尝试输入完整 DOI 或更准确的英文标题。", [], [], ""

    selected = matches[0]
    doi = selected.get("doi") or "未记录"
    info = [
        "## 已锁定论文",
        f"**{selected['title']}**",
        f"- DOI: `{doi}`",
        f"- 本地记录: `{selected.get('pdf_file') or '未记录'}`",
        "",
        "后续问题将优先检索这篇论文的实验、结果、结论和图表信息。",
    ]
    if selected.get("source_level") == "structured_only":
        info.extend([
            "",
            "> ⚠️ 当前仅有结构化实验数据，没有本地全文或 OCR。可以查询已入库的催化剂、"
            "产物、FE 和测试条件；涉及机理、图表或原文结论的问题需要先补充 PDF。",
        ])
    if len(matches) > 1:
        info.append("\n<details><summary>其他候选结果</summary>\n")
        for candidate in matches[1:]:
            info.append(
                f"- {candidate['title']} — DOI: {candidate.get('doi') or '未记录'}"
            )
        info.append("\n</details>")

    figures = []
    normalized_doi = str(doi).lower().strip()
    selected_pdf = str(selected.get("pdf_file") or "").lower().strip()
    if normalized_doi and normalized_doi != "未记录":
        for figure in load_figure_index():
            figure_doi = str(figure.get("doi", "")).lower().strip()
            figure_pdf = str(figure.get("pdf", "")).lower().strip()
            if figure_doi != normalized_doi and figure_pdf != selected_pdf:
                continue
            category = figure.get("category", "other")
            path = str(OUTPUT_DIR / "figures" / category / figure.get("file", ""))
            caption = (figure.get("caption", "") or "")[:180]
            figures.append((path, caption))
            if len(figures) >= 12:
                break

    return selected["index"], "\n".join(info), [], figures, ""


def add_single_paper_message(query: str, history: list, paper_index):
    """Add a question only when a paper has been selected."""
    if paper_index is None:
        history = history or []
        history = history + [{
            "role": "assistant",
            "content": "请先在上方输入 DOI 或论文标题并锁定一篇论文。",
        }]
        return history, "", "请先锁定论文。"
    if not query.strip():
        return history, "", ""
    history = (history or []) + [{"role": "user", "content": query}]
    return history, "", ""


def generate_single_paper_answer(history: list, paper_index):
    """Stream an answer grounded in the currently locked paper."""
    if paper_index is None or not history:
        yield history, "请先锁定论文。"
        return

    last = history[-1]
    query = _extract_text(last.get("content", "")) if isinstance(last, dict) else str(last)
    if not query.strip():
        yield history, ""
        return

    chat_pairs = []
    i = 0
    while i < len(history) - 1:
        current = history[i]
        following = history[i + 1] if i + 1 < len(history) else None
        if (
            isinstance(current, dict)
            and isinstance(following, dict)
            and current.get("role") == "user"
            and following.get("role") == "assistant"
        ):
            chat_pairs.append((
                _extract_text(current.get("content", "")),
                _extract_text(following.get("content", "")),
            ))
            i += 2
        else:
            i += 1

    try:
        for partial_answer, evidence in answer_single_paper_stream(
            query,
            int(paper_index),
            client=client,
            model=DEFAULT_MODEL,
            chat_history=chat_pairs,
        ):
            updated = history + [{"role": "assistant", "content": partial_answer}]
            yield updated, evidence
    except Exception as exc:
        updated = history + [{
            "role": "assistant",
            "content": f"单篇论文问答失败: {exc}",
        }]
        yield updated, str(exc)


def get_db_stats() -> str:
    """Return database statistics as markdown."""
    records = load_records()
    lines = [
        f"## 数据库统计\n",
        f"- 总记录数: **{len(records)}** 条",
        f"- 覆盖论文: **{records['doi'].nunique()}** 篇",
        f"- 年份范围: {int(records['year'].min())}-{int(records['year'].max())}",
        f"- 催化剂种类: {records['catalyst_name'].nunique()} 种",
        f"\n### 产物分布\n",
    ]
    product_stats = records.groupby("product").agg(
        count=("fe_pct", "size"),
        mean_fe=("fe_pct", "mean"),
        max_fe=("fe_pct", "max"),
    ).sort_values("count", ascending=False)

    lines.append("| 产物 | 记录数 | 平均FE% | 最高FE% |")
    lines.append("|------|--------|---------|---------|")
    for product, row in product_stats.iterrows():
        lines.append(f"| {product} | {int(row['count'])} | {row['mean_fe']:.1f}% | {row['max_fe']:.1f}% |")

    lines.append(f"\n### 催化剂类型分布\n")
    type_counts = records["catalyst_type"].value_counts().head(8)
    lines.append("| 类型 | 记录数 |")
    lines.append("|------|--------|")
    for t, c in type_counts.items():
        lines.append(f"| {t} | {c} |")

    # Synthesis method stats
    if "synthesis_method" in records.columns:
        syn_records = records[records["synthesis_method"].notna() & (records["synthesis_method"] != "")]
        lines.append(f"\n### 合成方法分布\n")
        lines.append(f"- 有合成方法数据的记录: **{len(syn_records)}** / {len(records)} ({100*len(syn_records)/len(records):.0f}%)")
        from collections import Counter
        all_methods = []
        for m in syn_records["synthesis_method"]:
            all_methods.extend(m.split("; "))
        method_counts = Counter(all_methods)
        lines.append("\n| 合成方法 | 记录数 |")
        lines.append("|----------|--------|")
        for method, count in method_counts.most_common(12):
            lines.append(f"| {method} | {count} |")

    return "\n".join(lines)


def _format_table_value(column: str, value) -> str:
    """Format one database value for the compact HTML results table."""
    if pd.isna(value) or value == "":
        return "—"
    if column == "year":
        return str(int(float(value)))
    if column in {"fe_pct", "voltage_V"}:
        number = float(value)
        return f"{number:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _render_database_table(records: pd.DataFrame, columns: list[str]) -> str:
    """Render one semantic table so header and body always share column widths."""
    labels = {
        "catalyst_name": "催化剂", "catalyst_type": "类型", "product": "产物",
        "fe_pct": "FE%", "voltage_V": "电位(V)", "electrolyte": "电解质",
        "synthesis_method": "合成方法", "doi": "DOI", "year": "年份",
    }
    widths = ["22%", "7%", "7%", "6%", "7%", "8%", "15%", "23%", "5%"]
    colgroup = "".join(f'<col style="width:{width}">' for width in widths)
    header = "".join(f"<th>{html.escape(labels[column])}</th>" for column in columns)
    rows = []
    for _, row in records.iterrows():
        cells = []
        for column in columns:
            text = _format_table_value(column, row[column])
            escaped = html.escape(text)
            if column == "doi" and text != "—":
                href = "https://doi.org/" + html.escape(text, quote=True)
                content = f'<a href="{href}" target="_blank">{escaped}</a>'
            else:
                content = escaped
            cells.append(
                f'<td class="col-{column}" title="{html.escape(text, quote=True)}">'
                f"{content}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    if not rows:
        rows.append(f'<tr><td colspan="{len(columns)}" class="empty-row">无匹配记录</td></tr>')
    return (
        '<div class="db-table-scroll"><table class="db-results-table">'
        f"<colgroup>{colgroup}</colgroup><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def search_database(product_filter: str, element_filter: str, min_fe: float) -> str:
    """Search and filter the structured database."""
    records = load_records().copy()

    # The database keeps the complete product distribution for modelling, but
    # this UI is a paper-level performance overview.  Select the representative
    # (highest-FE) product *before* applying the product filter, so a minor CO
    # coproduct cannot make a formate-selective catalyst appear in a CO search.
    group_cols = ["doi", "catalyst_name"]
    if "is_representative" in records.columns:
        representative = records["is_representative"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        group_has_representative = representative.groupby(
            [records[column] for column in group_cols], dropna=False
        ).transform("any")
        explicit = records[representative]
        fallback_pool = records[~group_has_representative]
        fallback = fallback_pool.loc[
            fallback_pool.groupby(group_cols, dropna=False)["fe_pct"].idxmax()
        ] if not fallback_pool.empty else fallback_pool
        records = pd.concat([explicit, fallback], ignore_index=True)
    else:
        records = records.loc[
            records.groupby(group_cols, dropna=False)["fe_pct"].idxmax()
        ]

    if product_filter and product_filter not in ("全部", "ALL"):
        records = records[records["product"] == product_filter]

    if element_filter and element_filter.strip():
        for elem in element_filter.split(","):
            elem = elem.strip()
            if elem:
                records = records[
                    records["catalyst_name"].str.contains(elem, case=True, na=False)
                ]

    if min_fe > 0:
        records = records[records["fe_pct"] >= min_fe]

    # Select display columns and sort
    display_cols = ["catalyst_name", "catalyst_type", "product", "fe_pct",
                    "voltage_V", "electrolyte", "synthesis_method", "doi", "year"]
    available_cols = [c for c in display_cols if c in records.columns]
    result = records[available_cols].sort_values("fe_pct", ascending=False).head(50)
    return _render_database_table(result, available_cols)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "推荐一个CO2到HCOOH的高效催化剂，合成要简单",
    "oxide-derived Cu 和普通Cu在C2产物上有什么差异？",
    "Cu-Sn合金催化剂主要生成什么产物？",
    "单原子Cu催化剂的研究进展如何？",
    "Cu基催化剂的XRD衍射峰一般是什么样的？",
    "Cu单原子催化剂的SEM/TEM形貌特征？",
    "一般CO2RR的电化学测试电位窗口是多少？",
    "有什么合成方法最广泛应用？",
]

# Figure category display names
FIGURE_CATEGORIES = {
    "xrd": "XRD 衍射",
    "electron_microscopy": "电子显微镜 (SEM/TEM)",
    "spectroscopy": "光谱学 (XPS/Raman)",
    "electrochemistry": "电化学测试",
    "faradaic_efficiency": "法拉第效率",
    "current_density": "电流密度",
    "stability": "稳定性测试",
    "morphology_scheme": "结构示意图",
    "other": "其他/未分类",
}


def browse_figures(category: str, element_filter: str) -> list:
    """Browse figures by category, return image paths for gallery."""
    fig_index = load_figure_index()
    if not fig_index:
        return []

    # Map display name back to category key
    cat_key = None
    for k, v in FIGURE_CATEGORIES.items():
        if v == category:
            cat_key = k
            break
    if cat_key is None:
        cat_key = category

    results = []
    for fig in fig_index:
        if fig.get("category") != cat_key:
            continue

        # Element filter
        if element_filter and element_filter.strip():
            caption = (fig.get("caption", "") or "").lower()
            pdf_name = fig.get("pdf", "").lower()
            if not any(
                e.strip().lower() in caption or e.strip().lower() in pdf_name
                for e in element_filter.split(",")
            ):
                continue

        # Build image path
        img_path = str(OUTPUT_DIR / "figures" / cat_key / fig["file"])
        caption = fig.get("caption", "")[:100] or fig.get("file", "")
        doi = fig.get("doi", "")
        label = f"{caption}\n(DOI: {doi})" if doi else caption
        results.append((img_path, label))

        if len(results) >= 20:
            break

    return results


# ---------------------------------------------------------------------------
# Administrator-only figure review
# ---------------------------------------------------------------------------

FIGURE_PILOT_PATH = OUTPUT_DIR / "figure_reclassification_pilot.json"
FIGURE_REVIEW_PATH = OUTPUT_DIR / "figure_human_reviews.csv"

REVIEW_CATEGORIES = {
    "xrd": "XRD 衍射",
    "electron_microscopy": "电子显微镜 SEM/TEM/STEM",
    "xps": "XPS",
    "raman_ftir": "Raman / FTIR",
    "xas": "XANES / EXAFS",
    "electrochemistry": "电化学动力学 CV/LSV/Tafel/EIS",
    "faradaic_efficiency": "法拉第效率 / 产物分布",
    "current_density": "电流密度",
    "stability": "稳定性测试",
    "theoretical_calculation": "理论计算 DFT / Free energy / PDOS",
    "thermal_analysis": "热分析 TGA / DSC",
    "adsorption_analysis": "吸附分析 BET / 孔径分布",
    "schematic": "合成 / 装置 / 机理示意图",
    "other": "其他 / 无法确定",
}
REVIEW_FILTERS = ["待人工审核", "类别变化", "图文冲突", "全部"]


def _admin_password() -> str:
    return os.getenv('CO2RR_ADMIN_PASSWORD', '').strip()


def _require_admin(password: str) -> None:
    expected = _admin_password()
    supplied = str(password or "")
    if not expected or not hmac.compare_digest(supplied, expected):
        raise gr.Error("管理员密码错误，或服务器尚未配置管理员密码。")


def _load_human_reviews() -> dict:
    if not FIGURE_REVIEW_PATH.exists():
        return {}
    with FIGURE_REVIEW_PATH.open(encoding="utf-8", newline="") as handle:
        return {
            row["file"]: row
            for row in csv.DictReader(handle)
            if row.get("file")
        }


def _load_pilot_rows(review_filter: str, unreviewed_only: bool) -> list:
    if not FIGURE_PILOT_PATH.exists():
        return []
    rows = json.loads(FIGURE_PILOT_PATH.read_text(encoding="utf-8"))
    reviews = _load_human_reviews()
    if review_filter == "待人工审核":
        rows = [row for row in rows if row.get("status") != "accepted"]
    elif review_filter == "类别变化":
        rows = [
            row for row in rows
            if row.get("old_category") != row.get("new_category")
        ]
    elif review_filter == "图文冲突":
        rows = [row for row in rows if row.get("caption_conflict")]
    if unreviewed_only:
        rows = [row for row in rows if row.get("file") not in reviews]
    return rows


def _review_markdown(row: dict, reviewed: dict) -> str:
    primary = row.get("primary") or {}
    secondary_model = row.get("review") or {}
    evidence = primary.get("visual_evidence", "未提供")
    axes = primary.get("visible_axes_text", "未识别")
    caption = row.get("caption") or "（无可靠图注）"
    review_line = "未审核"
    if reviewed:
        review_line = (
            f"{reviewed.get('decision', '')} / "
            f"{reviewed.get('human_primary_category', '')}"
        )
    secondary_summary = "未调用"
    if secondary_model:
        secondary_summary = (
            f"{secondary_model.get('primary_category', 'other')} "
            f"(模型置信度 {secondary_model.get('confidence', '未给出')})"
        )
    return f"""### 模型判断

- 文件：`{html.escape(str(row.get('file', '')))}`
- PDF：`{html.escape(str(row.get('pdf', '')))}`
- 旧标签：**{html.escape(str(row.get('old_category', '')))}**
- 新标签：**{html.escape(str(row.get('new_category', '')))}**
- 综合置信度：**{float(row.get('confidence', 0)):.3f}**
- 自动状态：**{html.escape(str(row.get('status', '')))}**
- 图文冲突：**{'是' if row.get('caption_conflict') else '否'}**
- 第二模型：{html.escape(secondary_summary)}
- 当前人工审核：**{html.escape(review_line)}**

**视觉证据**：{html.escape(str(evidence))}

**识别出的坐标轴/文字**：{html.escape(str(axes))}

**现有图注（仅作辅助）**：{html.escape(str(caption))}
"""


def _review_outputs(password: str, state: dict):
    _require_admin(password)
    rows = state.get("rows", []) if state else []
    position = int(state.get("position", 0)) if state else 0
    if not rows:
        empty_state = {"rows": [], "position": 0}
        return (
            empty_state, None, "没有符合当前筛选条件的图片。", None,
            [], "", "进度：0 / 0",
        )
    position = max(0, min(position, len(rows) - 1))
    state = {"rows": rows, "position": position}
    row = rows[position]
    reviewed = _load_human_reviews().get(row.get("file"), {})
    primary_value = reviewed.get("human_primary_category") or row.get("new_category")
    secondary_value = reviewed.get("human_secondary_categories", "")
    secondary_values = [value for value in secondary_value.split(";") if value]
    if not secondary_values:
        secondary_values = list((row.get("primary") or {}).get("secondary_categories") or [])
    note = reviewed.get("note", "")
    progress = (
        f"进度：{position + 1} / {len(rows)} ｜ "
        f"已审核总数：{len(_load_human_reviews())}"
    )
    return (
        state, row.get("path"), _review_markdown(row, reviewed),
        primary_value, secondary_values, note, progress,
    )


def load_figure_review(
    password: str, review_filter: str, unreviewed_only: bool
):
    _require_admin(password)
    rows = _load_pilot_rows(review_filter, unreviewed_only)
    return _review_outputs(password, {"rows": rows, "position": 0})


def move_figure_review(password: str, state: dict, step: int):
    _require_admin(password)
    state = dict(state or {"rows": [], "position": 0})
    state["position"] = int(state.get("position", 0)) + int(step)
    return _review_outputs(password, state)


def previous_figure_review(password: str, state: dict):
    return move_figure_review(password, state, -1)


def next_figure_review(password: str, state: dict):
    return move_figure_review(password, state, 1)


def _save_human_review(record: dict) -> None:
    reviews = _load_human_reviews()
    reviews[record["file"]] = record
    fields = [
        "file", "pdf", "old_category", "model_category", "model_confidence",
        "human_primary_category", "human_secondary_categories", "decision",
        "note", "reviewer", "review_time",
    ]
    FIGURE_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_REVIEW_PATH.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviews.values())
    temporary.replace(FIGURE_REVIEW_PATH)


def _save_and_next_figure_review(
    password: str,
    state: dict,
    human_primary: str,
    human_secondary: list,
    note: str,
    review_filter: str,
    unreviewed_only: bool,
    action: str,
):
    _require_admin(password)
    rows = state.get("rows", []) if state else []
    position = int(state.get("position", 0)) if state else 0
    if not rows:
        raise gr.Error("当前没有可保存的图片。")
    row = rows[max(0, min(position, len(rows) - 1))]
    if action == "exclude":
        human_primary = "other"
        human_secondary = []
        decision = "排除"
    elif action == "uncertain":
        human_primary = "other"
        human_secondary = []
        decision = "无法判断"
    else:
        if not human_primary:
            raise gr.Error("请选择人工主类别。")
        model_primary = row.get("new_category", "other")
        model_secondary = set(
            (row.get("primary") or {}).get("secondary_categories") or []
        )
        human_secondary_set = set(human_secondary or [])
        if (
            human_primary == model_primary
            and human_secondary_set == model_secondary
        ):
            decision = "人工确认（与模型一致）"
        else:
            decision = "人工修正"
    _save_human_review({
        "file": row.get("file", ""),
        "pdf": row.get("pdf", ""),
        "old_category": row.get("old_category", ""),
        "model_category": row.get("new_category", ""),
        "model_confidence": row.get("confidence", 0),
        "human_primary_category": human_primary or row.get("new_category", "other"),
        "human_secondary_categories": ";".join(human_secondary or []),
        "decision": decision,
        "note": str(note or "").strip(),
        "reviewer": "admin",
        "review_time": datetime.now().isoformat(timespec="seconds"),
    })
    refreshed = _load_pilot_rows(review_filter, unreviewed_only)
    next_position = min(position, max(0, len(refreshed) - 1))
    return _review_outputs(
        password, {"rows": refreshed, "position": next_position}
    )


def save_and_next_figure_review(
    password: str,
    state: dict,
    human_primary: str,
    human_secondary: list,
    note: str,
    review_filter: str,
    unreviewed_only: bool,
):
    return _save_and_next_figure_review(
        password, state, human_primary, human_secondary, note,
        review_filter, unreviewed_only, "save",
    )


def mark_uncertain_figure_review(
    password: str,
    state: dict,
    human_primary: str,
    human_secondary: list,
    note: str,
    review_filter: str,
    unreviewed_only: bool,
):
    return _save_and_next_figure_review(
        password, state, human_primary, human_secondary, note,
        review_filter, unreviewed_only, "uncertain",
    )


def exclude_figure_review(
    password: str,
    state: dict,
    human_primary: str,
    human_secondary: list,
    note: str,
    review_filter: str,
    unreviewed_only: bool,
):
    return _save_and_next_figure_review(
        password, state, human_primary, human_secondary, note,
        review_filter, unreviewed_only, "exclude",
    )


CUSTOM_CSS = """
/* ===== Apple-inspired CO2RR Knowledge Base Theme ===== */

/* Global font & color */
:root {
    --bg-primary: #fafafa;
    --bg-card: #ffffff;
    --bg-hero: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 40%, #0d2137 70%, #0a2a1a 100%);
    --text-primary: #1d1d1f;
    --text-secondary: #6e6e73;
    --text-hero: #f5f5f7;
    --accent: #0071e3;
    --accent-hover: #0077ed;
    --accent-green: #30d158;
    --border: rgba(0,0,0,0.06);
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.08);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.1);
    --radius: 16px;
    --radius-sm: 10px;
}

/* Dark mode overrides */
.dark, .dark :root {
    --bg-primary: #0d0d0d;
    --bg-card: #1c1c1e;
    --text-primary: #f5f5f7;
    --text-secondary: #a1a1a6;
    --accent: #2997ff;
    --accent-hover: #40a9ff;
    --accent-green: #30d158;
    --border: rgba(255,255,255,0.1);
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.3);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.4);
}
.dark .gradio-container {
    background: var(--bg-primary) !important;
}
.dark .tabs > .tab-nav > button {
    color: var(--text-secondary) !important;
}
.dark .tabs > .tab-nav > button.selected {
    color: var(--text-primary) !important;
    border-color: var(--accent) !important;
}
.dark .prose, .dark .prose p, .dark .prose li, .dark .prose td, .dark .prose th {
    color: var(--text-primary) !important;
}
.dark .prose h2, .dark .prose h3, .dark .prose strong {
    color: #ffffff !important;
}
.dark .examples-row button {
    background: var(--bg-card) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: var(--text-secondary) !important;
}
.dark .examples-row button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
.dark .markdown-text, .dark .md, .dark p, .dark label, .dark span.text-lg {
    color: var(--text-primary) !important;
}
.dark textarea, .dark input[type="text"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-color: rgba(255,255,255,0.12) !important;
}
.dark .accordion {
    background: var(--bg-card) !important;
    border-color: rgba(255,255,255,0.08) !important;
}
.dark table, .dark table th, .dark table td {
    border-color: rgba(255,255,255,0.1) !important;
}
.dark table th {
    background: rgba(255,255,255,0.05) !important;
    color: var(--text-primary) !important;
}

/* Base */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                 "Helvetica Neue", Arial, sans-serif !important;
    background: var(--bg-primary) !important;
    max-width: 1760px !important;
    width: min(96vw, 1760px) !important;
    margin: 0 auto !important;
}

/* Hero section */
#hero-section {
    background: linear-gradient(135deg, #0a0a1a 0%, #12122e 30%, #0d2137 60%, #0a2a1a 100%);
    border-radius: var(--radius);
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    align-items: center;
    gap: 48px;
    min-height: 276px;
}
/* Orbital glow top-right */
#hero-section::before {
    content: "";
    position: absolute;
    top: -120px; right: -80px;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(0,113,227,0.18) 0%, rgba(41,151,255,0.06) 40%, transparent 70%);
    pointer-events: none;
    animation: pulse-glow 6s ease-in-out infinite alternate;
}
/* Green glow bottom-left */
#hero-section::after {
    content: "";
    position: absolute;
    bottom: -100px; left: -60px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(48,209,88,0.12) 0%, transparent 65%);
    pointer-events: none;
    animation: pulse-glow 8s ease-in-out 2s infinite alternate;
}
@keyframes pulse-glow {
    0% { opacity: 0.6; transform: scale(1); }
    100% { opacity: 1; transform: scale(1.08); }
}

/* Left: text content */
.hero-left {
    position: relative;
    z-index: 1;
    width: 100%;
}
#hero-section h1 {
    font-size: 2em !important;
    font-weight: 700 !important;
    color: #f5f5f7 !important;
    letter-spacing: -0.025em !important;
    margin: 0 0 6px 0 !important;
    line-height: 1.2 !important;
    white-space: nowrap;
}
#hero-section h1 sub {
    font-size: 0.6em;
    vertical-align: baseline;
    position: relative;
    top: 0.25em;
}
#hero-section .hero-subtitle {
    color: rgba(245,245,247,0.55) !important;
    font-size: 0.92em !important;
    line-height: 1.55 !important;
    margin: 0 0 14px 0;
    max-width: 520px;
}
.hero-badges {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    width: 100%;
    max-width: 470px;
}
.hero-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    min-height: 40px;
    padding: 7px 12px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 500;
    letter-spacing: 0.01em;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
.badge-blue   { background: rgba(0,113,227,0.2);  color: #64d2ff; border: 1px solid rgba(100,210,255,0.15); }
.badge-green  { background: rgba(48,209,88,0.15); color: #30d158; border: 1px solid rgba(48,209,88,0.15); }
.badge-purple { background: rgba(191,90,242,0.15); color: #bf5af2; border: 1px solid rgba(191,90,242,0.15); }
.badge-orange { background: rgba(255,159,10,0.15); color: #ff9f0a; border: 1px solid rgba(255,159,10,0.15); }

/* Right: stat cards */
.hero-stats {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    position: relative;
    z-index: 1;
    width: 100%;
    padding-left: 48px;
    border-left: 1px solid rgba(255,255,255,0.1);
}
.stat-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    min-width: 0;
    min-height: 92px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: transform 0.2s, background 0.2s;
}
.stat-card:hover {
    background: rgba(255,255,255,0.1);
    transform: translateY(-2px);
}
.stat-number {
    font-size: 1.5em;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 2px;
}
.stat-label {
    font-size: 0.7em;
    color: rgba(245,245,247,0.5);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}
.stat-blue   .stat-number { color: #64d2ff; }
.stat-green  .stat-number { color: #30d158; }
.stat-purple .stat-number { color: #bf5af2; }
.stat-orange .stat-number { color: #ff9f0a; }

/* Molecular decoration dots */
.hero-molecule {
    position: absolute;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
}
.mol-1 { width: 6px; height: 6px; background: rgba(100,210,255,0.35); top: 18%; right: 38%; }
.mol-2 { width: 4px; height: 4px; background: rgba(48,209,88,0.3);  top: 72%; right: 52%; }
.mol-3 { width: 8px; height: 8px; background: rgba(191,90,242,0.2); top: 28%; right: 12%; }
.mol-4 { width: 5px; height: 5px; background: rgba(255,159,10,0.25); top: 65%; right: 22%; }
.mol-5 { width: 3px; height: 3px; background: rgba(100,210,255,0.2); top: 45%; right: 45%; }
/* Connecting lines (pseudo-bonds) */
.hero-bond {
    position: absolute;
    height: 1px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.12;
}
.bond-1 { width: 60px; background: linear-gradient(90deg, #64d2ff, transparent); top: 22%; right: 35%; transform: rotate(-25deg); }
.bond-2 { width: 45px; background: linear-gradient(90deg, transparent, #30d158); top: 68%; right: 48%; transform: rotate(15deg); }
.bond-3 { width: 55px; background: linear-gradient(90deg, #bf5af2, transparent); top: 32%; right: 18%; transform: rotate(-40deg); }

@media (max-width: 900px) {
    #hero-section {
        grid-template-columns: 1fr;
        gap: 26px;
        padding: 30px 28px;
    }
    #hero-section h1 {
        white-space: normal;
    }
    .hero-subtitle,
    .hero-badges {
        max-width: none !important;
    }
    .hero-stats {
        padding-left: 0;
        padding-top: 26px;
        border-left: 0;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
}

@media (max-width: 520px) {
    .hero-badges,
    .hero-stats {
        grid-template-columns: 1fr;
    }
}

/* Tabs */
.tabs > .tab-nav {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
    padding: 4px !important;
    box-shadow: var(--shadow-sm) !important;
    border: none !important;
    margin-bottom: 16px !important;
}
.tabs > .tab-nav > button {
    font-family: inherit !important;
    font-weight: 500 !important;
    font-size: 0.92em !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    color: var(--text-secondary) !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.tabs > .tab-nav > button.selected {
    background: var(--accent) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(0,113,227,0.3) !important;
}

/* Cards / panels */
.block, .form, .panel {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Chatbot */
.chatbot {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-md) !important;
}
.chatbot .message {
    font-family: inherit !important;
    line-height: 1.6 !important;
}

/* Buttons */
button.primary {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95em !important;
    letter-spacing: 0.01em !important;
    padding: 10px 24px !important;
    box-shadow: 0 2px 8px rgba(0,113,227,0.25) !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 4px 12px rgba(0,113,227,0.35) !important;
    transform: translateY(-1px);
}

/* Textbox */
textarea, input[type="text"] {
    font-family: inherit !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.15) !important;
}

/* Accordion */
.accordion {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}

/* Gallery */
.gallery {
    border-radius: var(--radius-sm) !important;
}
.gallery .thumbnail-item {
    border-radius: 8px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.gallery .thumbnail-item:hover {
    transform: scale(1.02);
    box-shadow: var(--shadow-md) !important;
}

/* Dataframe */
table {
    font-family: inherit !important;
    font-size: 0.88em !important;
}
table th {
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

/* One native table (instead of Gradio's split virtual header/body) guarantees
   exact alignment and keeps the year column visible. */
.db-table-scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }
.db-results-table { width: 100%; min-width: 1420px; table-layout: fixed; border-collapse: collapse; font-size: 14px !important; }
.db-results-table th, .db-results-table td { box-sizing: border-box; padding: 10px 12px !important; line-height: 1.3; text-align: left; vertical-align: middle; white-space: nowrap; border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); overflow: hidden; text-overflow: ellipsis; }
.db-results-table th { font-weight: 650 !important; background: var(--bg-card); }
.db-results-table th:last-child, .db-results-table td:last-child { border-right: 0; }
.db-results-table tbody tr:last-child td { border-bottom: 0; }
.db-results-table tbody tr:nth-child(even) { background: rgba(127,127,127,0.035); }
.db-results-table .col-doi { font-size: 13px; }
.db-results-table .col-year { min-width: 72px; overflow: visible; text-overflow: clip; }
.db-results-table a { color: var(--accent); text-decoration: none; }
.db-results-table a:hover { text-decoration: underline; }
.db-results-table .empty-row { text-align: center; color: var(--text-secondary); padding: 28px !important; }

/* Examples */
.examples-row {
    gap: 8px !important;
}
.examples-row button {
    border-radius: 20px !important;
    font-size: 0.85em !important;
    padding: 6px 16px !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    background: var(--bg-card) !important;
    transition: all 0.15s !important;
}
.examples-row button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Stats markdown */
.prose h2 {
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}
.prose table {
    border-radius: 8px !important;
    overflow: hidden;
}

/* Slider */
input[type="range"] {
    accent-color: var(--accent) !important;
}

/* Dropdown */
.dropdown {
    border-radius: var(--radius-sm) !important;
}
"""

HERO_HTML = """
<div id="hero-section">
  <!-- Molecular decoration -->
  <div class="hero-molecule mol-1"></div>
  <div class="hero-molecule mol-2"></div>
  <div class="hero-molecule mol-3"></div>
  <div class="hero-molecule mol-4"></div>
  <div class="hero-molecule mol-5"></div>
  <div class="hero-bond bond-1"></div>
  <div class="hero-bond bond-2"></div>
  <div class="hero-bond bond-3"></div>

  <div class="hero-left">
    <h1>CO<sub>2</sub>RR&nbsp;Catalyst&nbsp;Intelligence</h1>
    <p class="hero-subtitle">
      Retrieval-augmented knowledge base for CO<sub>2</sub> electroreduction research &mdash;
      combining structured data, full-text literature, and characterization analysis.
    </p>
    <div class="hero-badges">
      <span class="hero-badge badge-blue">&#x2728; DeepSeek LLM</span>
      <span class="hero-badge badge-green">&#x1F50D; RAG Retrieval</span>
      <span class="hero-badge badge-purple">&#x1F52C; XRD / SEM / XPS</span>
      <span class="hero-badge badge-orange">&#x26A1; FE% Database</span>
    </div>
  </div>

  <div class="hero-stats">
    <div class="stat-card stat-blue">
      <div class="stat-number">661</div>
      <div class="stat-label">Papers</div>
    </div>
    <div class="stat-card stat-green">
      <div class="stat-number">813</div>
      <div class="stat-label">Records</div>
    </div>
    <div class="stat-card stat-purple">
      <div class="stat-number">3k+</div>
      <div class="stat-label">Figures</div>
    </div>
    <div class="stat-card stat-orange">
      <div class="stat-number">516</div>
      <div class="stat-label">PDFs</div>
    </div>
  </div>
</div>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="CO2RR Catalyst Intelligence") as app:
        hero = HERO_HTML.replace('>661<', f'>{len(load_knowledge_base())}<')
        hero = hero.replace('>813<', f'>{len(df)}<')
        hero = hero.replace('>3k+<', f'>{len(load_figure_index())}<')
        pdf_count = len(list((PROJECT_ROOT / 'gold_corpus' / 'pdfs').glob('*.pdf')))
        hero = hero.replace('>516<', f'>{pdf_count}<')
        gr.HTML(hero)
        gr.Markdown('学生调试版：默认仅包含少量示例记录，不能代表完整文献数据库。')
        if client is None:
            gr.Markdown('未配置 DEEPSEEK_API_KEY：可浏览数据库与检索示例；AI 回答需要配置密钥并重启。')

        with gr.Tabs():
            # ---- Tab 1: Q&A ----
            with gr.Tab("Ask Knowledge Base"):
                chatbot = gr.Chatbot(
                    label="",
                    height=460,
                    placeholder="Ask anything about CO2RR catalysts...",
                )
                with gr.Row():
                    query_input = gr.Textbox(
                        show_label=False,
                        placeholder="e.g.  Cu-Sn alloy for HCOOH / XRD peaks of Cu nanoparticles / best catalyst for C2H4",
                        scale=5,
                        lines=1,
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)

                with gr.Accordion("Retrieval details", open=False):
                    debug_output = gr.Textbox(label="", lines=5, interactive=False)

                gr.Markdown("**Try these:**")
                examples = gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUERIES],
                    inputs=[query_input],
                )

                # 2-step event: show user msg immediately, then generate
                submit_btn.click(
                    fn=add_user_message,
                    inputs=[query_input, chatbot],
                    outputs=[chatbot, debug_output, query_input],
                ).then(
                    fn=generate_answer,
                    inputs=[chatbot],
                    outputs=[chatbot, debug_output],
                )
                query_input.submit(
                    fn=add_user_message,
                    inputs=[query_input, chatbot],
                    outputs=[chatbot, debug_output, query_input],
                ).then(
                    fn=generate_answer,
                    inputs=[chatbot],
                    outputs=[chatbot, debug_output],
                )

            # ---- Tab 2: Single-paper deep Q&A ----
            with gr.Tab("Single Paper Q&A"):
                gr.Markdown(
                    "输入 DOI 或英文标题锁定一篇论文，然后围绕实验方法、测试条件、"
                    "图表、机理、结论与局限性连续追问。"
                )
                selected_paper_index = gr.State(value=None)
                with gr.Row():
                    paper_search = gr.Textbox(
                        label="DOI or paper title",
                        placeholder="e.g. 10.1021/jacs.0c06420",
                        scale=5,
                    )
                    paper_select_btn = gr.Button("Lock Paper", variant="primary", scale=1)

                paper_info = gr.Markdown("尚未锁定论文。")
                paper_gallery = gr.Gallery(
                    label="Figures from this paper",
                    columns=4,
                    height=260,
                    object_fit="contain",
                )
                paper_chatbot = gr.Chatbot(
                    label="Single-paper discussion",
                    height=430,
                    placeholder="Lock a paper, then ask about methods, data, figures, mechanisms, or limitations.",
                )
                with gr.Row():
                    paper_question = gr.Textbox(
                        show_label=False,
                        placeholder="例如：83.2%的乙烯FE在什么条件下获得？证据在哪一段？",
                        scale=5,
                        lines=1,
                    )
                    paper_ask_btn = gr.Button("Ask Paper", variant="primary", scale=1)

                with gr.Accordion("Evidence used", open=False):
                    paper_evidence = gr.Textbox(label="", lines=12, interactive=False)

                gr.Examples(
                    examples=[
                        ["催化剂是如何合成的？请列出关键步骤和条件。"],
                        ["最高FE在什么电位、电解质和电流密度下获得？"],
                        ["作者提出了什么反应机理？证据是什么？"],
                        ["这篇论文的主要结论和局限性是什么？"],
                        ["把这篇论文与数据库中的相似催化剂进行比较。"],
                    ],
                    inputs=[paper_question],
                )

                paper_select_btn.click(
                    fn=select_single_paper,
                    inputs=[paper_search],
                    outputs=[
                        selected_paper_index,
                        paper_info,
                        paper_chatbot,
                        paper_gallery,
                        paper_evidence,
                    ],
                )
                paper_search.submit(
                    fn=select_single_paper,
                    inputs=[paper_search],
                    outputs=[
                        selected_paper_index,
                        paper_info,
                        paper_chatbot,
                        paper_gallery,
                        paper_evidence,
                    ],
                )

                paper_ask_btn.click(
                    fn=add_single_paper_message,
                    inputs=[paper_question, paper_chatbot, selected_paper_index],
                    outputs=[paper_chatbot, paper_question, paper_evidence],
                ).then(
                    fn=generate_single_paper_answer,
                    inputs=[paper_chatbot, selected_paper_index],
                    outputs=[paper_chatbot, paper_evidence],
                )
                paper_question.submit(
                    fn=add_single_paper_message,
                    inputs=[paper_question, paper_chatbot, selected_paper_index],
                    outputs=[paper_chatbot, paper_question, paper_evidence],
                ).then(
                    fn=generate_single_paper_answer,
                    inputs=[paper_chatbot, selected_paper_index],
                    outputs=[paper_chatbot, paper_evidence],
                )

            # ---- Tab 3: Database Explorer ----
            with gr.Tab("Catalyst Design Agent"):
                gr.Markdown(
                    "将设计目标转化为数据库约束，检索可追溯证据，并生成三个需要人工批准的"
                    "催化剂设计假设。该模块不会把候选方案或目标值表述为已经验证的性能。"
                )
                design_objective = gr.Textbox(
                    label="Research objective",
                    placeholder=(
                        "例如：设计一种以Cu为主、用于流动池制乙烯的低成本催化剂，"
                        "重点改善C–C偶联和50小时稳定性"
                    ),
                    lines=3,
                )
                with gr.Row(equal_height=True):
                    design_product = gr.Dropdown(
                        choices=["HCOOH", "CO", "C2H4", "C2H5OH", "CH4", "CH3OH", "C2H6"],
                        value="C2H4",
                        label="Target product",
                    )
                    design_required_elements = gr.Textbox(
                        value="Cu",
                        label="Required elements",
                        placeholder="例如：Cu 或 Cu,Ag",
                    )
                    design_excluded_elements = gr.Textbox(
                        label="Excluded elements",
                        placeholder="例如：Au,Pd,Pt",
                    )
                    design_cell = gr.Dropdown(
                        choices=["不限", "H-cell", "Flow cell", "MEA"],
                        value="不限",
                        label="Cell type",
                    )
                with gr.Row(equal_height=True):
                    design_min_fe = gr.Slider(
                        0, 100, value=50, step=1, label="Target minimum FE%"
                    )
                    design_min_current = gr.Number(
                        value=0,
                        minimum=0,
                        label="Target current density (mA cm⁻²; 0 =不限)",
                    )
                    design_synthesis = gr.Textbox(
                        label="Synthesis preference",
                        placeholder="例如：electrodeposition；留空表示不限",
                    )
                    design_btn = gr.Button(
                        "Retrieve Evidence & Design", variant="primary"
                    )

                design_output = gr.Markdown(
                    "设置目标与约束后启动。建议先把目标限定为一个产物和一种电解槽。"
                )
                design_task_id = gr.Textbox(
                    label="Current design task ID",
                    placeholder="生成设计后自动填写；也可粘贴历史 DESIGN-... 编号",
                    interactive=True,
                )
                with gr.Accordion("Design evidence and audit trail", open=False):
                    design_evidence = gr.Textbox(
                        label="",
                        lines=18,
                        interactive=False,
                    )

                design_btn.click(
                    fn=generate_and_save_catalyst_design,
                    inputs=[
                        design_objective,
                        design_product,
                        design_required_elements,
                        design_excluded_elements,
                        design_cell,
                        design_min_fe,
                        design_min_current,
                        design_synthesis,
                    ],
                    outputs=[
                        design_output,
                        design_evidence,
                        design_task_id,
                    ],
                )

                gr.Markdown("## Experiment Feedback / 实验结果回填")
                gr.Markdown(
                    "填写实际实验条件和结果。程序会先校验FE闭合、计算目标产物分电流密度，"
                    "再把实验事实作为 `[X#]` 证据交给模型。建议同时填写对照组并完成至少3次独立重复。"
                )
                recent_designs = gr.Markdown(recent_design_sessions())
                refresh_designs_btn = gr.Button("Refresh recent design tasks")
                refresh_designs_btn.click(
                    fn=recent_design_sessions,
                    outputs=[recent_designs],
                )

                with gr.Row(equal_height=True):
                    feedback_sample_id = gr.Textbox(
                        label="Sample ID",
                        placeholder="例如：CuAg-05-R1",
                    )
                    feedback_catalyst = gr.Textbox(
                        label="Actual catalyst / composition",
                        placeholder="填写实际制得的材料和比例",
                    )
                    feedback_cell = gr.Dropdown(
                        choices=["H-cell", "Flow cell", "MEA", "其他"],
                        value="H-cell",
                        label="Actual cell type",
                    )
                    feedback_synthesis_success = gr.Radio(
                        choices=["是", "部分成功", "否", "尚未确认"],
                        value="尚未确认",
                        label="Synthesis successful?",
                    )
                with gr.Row(equal_height=True):
                    feedback_potential = gr.Number(
                        value=None,
                        label="Potential (V)",
                    )
                    feedback_reference = gr.Textbox(
                        label="Reference electrode",
                        placeholder="例如：RHE",
                    )
                    feedback_electrolyte = gr.Textbox(
                        label="Electrolyte",
                        placeholder="例如：1 M KOH",
                    )
                    feedback_stability = gr.Number(
                        value=None,
                        minimum=0,
                        label="Stability test (h)",
                    )
                with gr.Row(equal_height=True):
                    feedback_total_current = gr.Number(
                        value=None,
                        minimum=0,
                        label="Total current density (mA cm⁻²)",
                    )
                    feedback_target_fe = gr.Number(
                        value=None,
                        minimum=0,
                        maximum=100,
                        label="Target-product FE%",
                    )
                    feedback_other_fe = gr.Number(
                        value=0,
                        minimum=0,
                        maximum=100,
                        label="All other products FE% (sum)",
                    )
                    feedback_replicates = gr.Number(
                        value=1,
                        minimum=1,
                        precision=0,
                        label="Independent replicates",
                    )
                with gr.Row(equal_height=True):
                    feedback_control_fe = gr.Number(
                        value=None,
                        minimum=0,
                        maximum=100,
                        label="Control target-product FE%",
                    )
                    feedback_control_current = gr.Number(
                        value=None,
                        minimum=0,
                        label="Control current density (mA cm⁻²)",
                    )
                    feedback_files = gr.File(
                        label="Raw data / characterization attachments",
                        file_count="multiple",
                        type="filepath",
                    )
                feedback_observations = gr.Textbox(
                    label="Experimental observations and deviations from the design",
                    placeholder=(
                        "记录颜色、沉积状态、盐析/淹没、失活、合成参数变化、"
                        "XRD/XPS初步观察以及异常情况"
                    ),
                    lines=4,
                )
                feedback_submit_btn = gr.Button(
                    "Save Feedback & Propose Next Round", variant="primary"
                )
                feedback_output = gr.Markdown("尚未提交实验反馈。")
                with gr.Accordion("Experimental evidence and validation audit", open=False):
                    feedback_audit = gr.Textbox(
                        label="",
                        lines=16,
                        interactive=False,
                    )

                feedback_submit_btn.click(
                    fn=analyze_experiment_feedback,
                    inputs=[
                        design_task_id,
                        feedback_sample_id,
                        feedback_catalyst,
                        feedback_cell,
                        feedback_potential,
                        feedback_reference,
                        feedback_electrolyte,
                        feedback_total_current,
                        feedback_target_fe,
                        feedback_other_fe,
                        feedback_control_fe,
                        feedback_control_current,
                        feedback_stability,
                        feedback_replicates,
                        feedback_synthesis_success,
                        feedback_observations,
                        feedback_files,
                    ],
                    outputs=[feedback_output, feedback_audit, recent_designs],
                )

            # ---- Tab 4: Database Explorer ----
            with gr.Tab("Explore Database"):
                with gr.Row(equal_height=True):
                    product_dropdown = gr.Dropdown(
                        choices=["ALL", "CO", "HCOOH", "C2H4", "C2H5OH", "CH4", "CH3OH", "C2H6", "syngas"],
                        value="ALL",
                        label="Product",
                    )
                    element_input = gr.Textbox(
                        label="Element filter",
                        placeholder="e.g. Sn or Bi,Sn",
                    )
                    min_fe_slider = gr.Slider(0, 100, value=0, step=5, label="Min FE%")
                    search_btn = gr.Button("Search", variant="primary")

                gr.Markdown("### Results (top 50 by FE%)")
                results_table = gr.HTML(
                    value=search_database("ALL", "", 0),
                    elem_id="results-table",
                )
                search_btn.click(
                    fn=search_database,
                    inputs=[product_dropdown, element_input, min_fe_slider],
                    outputs=[results_table],
                )

            # ---- Tab 5: Figure Browser ----
            with gr.Tab("Characterization Gallery"):
                with gr.Row(equal_height=True):
                    fig_category = gr.Dropdown(
                        choices=list(FIGURE_CATEGORIES.values()),
                        value="XRD 衍射",
                        label="Category",
                    )
                    fig_element = gr.Textbox(
                        label="Element (optional)",
                        placeholder="e.g. Cu",
                    )
                    fig_search_btn = gr.Button("Browse", variant="primary")

                fig_gallery = gr.Gallery(
                    label="",
                    columns=3,
                    height=520,
                    object_fit="contain",
                )
                fig_search_btn.click(
                    fn=browse_figures,
                    inputs=[fig_category, fig_element],
                    outputs=[fig_gallery],
                )

            # ---- Tab 6: Administrator figure review ----
            with gr.Tab("Figure Review (Admin)"):
                gr.Markdown(
                    "该页面仅供管理员审核视觉模型分类。所有操作均在服务器端验证密码；"
                    "审核结果单独保存，不会直接覆盖正式图库。"
                )
                review_state = gr.State(value={"rows": [], "position": 0})
                with gr.Row(equal_height=True):
                    review_password = gr.Textbox(
                        label="Admin password",
                        type="password",
                        placeholder="输入管理员密码",
                        scale=2,
                    )
                    review_filter = gr.Dropdown(
                        choices=REVIEW_FILTERS,
                        value="待人工审核",
                        label="审核范围",
                        scale=2,
                    )
                    unreviewed_only = gr.Checkbox(
                        value=True,
                        label="只显示未审核",
                        scale=1,
                    )
                    load_review_btn = gr.Button(
                        "Unlock / Refresh", variant="primary", scale=1
                    )

                review_progress = gr.Markdown("进度：0 / 0")
                with gr.Row(equal_height=False):
                    review_image = gr.Image(
                        label="当前图片",
                        type="filepath",
                        height=560,
                        scale=3,
                    )
                    with gr.Column(scale=2):
                        review_info = gr.Markdown("输入管理员密码后载入审核任务。")
                        human_primary = gr.Dropdown(
                            choices=[
                                (display, key)
                                for key, display in REVIEW_CATEGORIES.items()
                            ],
                            label="人工主类别",
                        )
                        human_secondary = gr.CheckboxGroup(
                            choices=[
                                (display, key)
                                for key, display in REVIEW_CATEGORIES.items()
                            ],
                            label="人工次类别（可多选）",
                        )
                        review_note = gr.Textbox(
                            label="审核备注",
                            placeholder="说明修改依据、冲突面板或排除原因",
                            lines=3,
                        )

                review_outputs = [
                    review_state,
                    review_image,
                    review_info,
                    human_primary,
                    human_secondary,
                    review_note,
                    review_progress,
                ]
                with gr.Row():
                    previous_review_btn = gr.Button("← Previous")
                    save_review_btn = gr.Button(
                        "保存分类并下一张", variant="primary"
                    )
                    uncertain_review_btn = gr.Button("无法判断")
                    exclude_review_btn = gr.Button("排除此图", variant="stop")
                    next_review_btn = gr.Button("Next →")

                load_review_btn.click(
                    fn=load_figure_review,
                    inputs=[review_password, review_filter, unreviewed_only],
                    outputs=review_outputs,
                )
                previous_review_btn.click(
                    fn=previous_figure_review,
                    inputs=[review_password, review_state],
                    outputs=review_outputs,
                )
                next_review_btn.click(
                    fn=next_figure_review,
                    inputs=[review_password, review_state],
                    outputs=review_outputs,
                )
                save_review_btn.click(
                    fn=save_and_next_figure_review,
                    inputs=[
                        review_password,
                        review_state,
                        human_primary,
                        human_secondary,
                        review_note,
                        review_filter,
                        unreviewed_only,
                    ],
                    outputs=review_outputs,
                )
                uncertain_review_btn.click(
                    fn=mark_uncertain_figure_review,
                    inputs=[
                        review_password,
                        review_state,
                        human_primary,
                        human_secondary,
                        review_note,
                        review_filter,
                        unreviewed_only,
                    ],
                    outputs=review_outputs,
                )
                exclude_review_btn.click(
                    fn=exclude_figure_review,
                    inputs=[
                        review_password,
                        review_state,
                        human_primary,
                        human_secondary,
                        review_note,
                        review_filter,
                        unreviewed_only,
                    ],
                    outputs=review_outputs,
                )

            # ---- Tab 7: Statistics ----
            with gr.Tab("Statistics"):
                fig_index = load_figure_index()
                fig_stats = ""
                if fig_index:
                    from collections import Counter
                    cat_counts = Counter(f.get("category", "other") for f in fig_index)
                    fig_stats = f"\n\n## Characterization Figures\n\n- Total: **{len(fig_index)}** images\n\n"
                    fig_stats += "| Category | Count |\n|----------|-------|\n"
                    for cat, name in FIGURE_CATEGORIES.items():
                        fig_stats += f"| {name} | {cat_counts.get(cat, 0)} |\n"

                stats_md = gr.Markdown(get_db_stats() + fig_stats)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CO2RR Knowledge Base Web UI")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind. If omitted, Gradio will search from 7860 upward.",
    )
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
