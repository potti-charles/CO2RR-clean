"""Evidence-grounded, human-in-the-loop catalyst design workflow.

The module deliberately separates deterministic database filtering from LLM
reasoning.  The generated proposals are hypotheses for experimental review,
not autonomous claims of catalyst performance.
"""

import math
import re
from typing import Iterable

import pandas as pd

from co2rr_knowledge_query import (
    DEFAULT_MODEL,
    detect_query_intent,
    load_records,
    retrieve_literature,
)


DESIGN_SYSTEM_PROMPT = """你是 CO2 电还原催化剂设计智能体，但你的权限仅限于提出可验证的科研假设。

必须遵守：
1. 只能把给出的 [E#] 数据记录和 [L#] 文献片段称为证据，不得编造 DOI、数值、物相或合成条件。
2. 不得把 C2/C2+ 总 FE 当作 C2H4 FE，不得跨 H-cell、flow cell、MEA 直接宣称性能优劣。
3. 新候选只能称为“设计假设”，不得给出没有模型依据的精确预测 FE；只描述预期变化方向及不确定性。
4. 明确区分：文献事实、机理推断、待验证假设。每项关键事实用 [E#] 或 [L#] 标注。
5. 若证据不足或字段缺失，必须指出，不得补写看似合理的数字。
6. 候选必须满足禁用元素等硬约束，并说明工作电位下重构、溶出、传质和竞争析氢等风险。
7. 输出实验方案必须包含对照组、重复实验、产物定量、总 FE 闭合检查及表征手段。
8. 用中文回答，先给结论；最终选择权属于研究人员。
"""


def _tokens(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，;；\s]+", value or "") if item.strip()]


def _contains_element(name: str, element: str) -> bool:
    """Conservatively match an element symbol or common English name."""
    aliases = {
        "Cu": "copper", "Sn": "tin", "Ag": "silver", "Bi": "bismuth",
        "In": "indium", "Au": "gold", "Zn": "zinc", "Pd": "palladium",
        "Ni": "nickel", "Fe": "iron", "Co": "cobalt", "Pt": "platinum",
    }
    text = str(name or "")
    symbols = set(re.findall(r"[A-Z][a-z]?", text))
    canonical = element[:1].upper() + element[1:].lower()
    if canonical in symbols:
        return True
    alias = aliases.get(canonical, element).lower()
    return bool(re.search(rf"\b{re.escape(alias)}\b", text.lower()))


def _representative_rows(records: pd.DataFrame) -> pd.DataFrame:
    """Keep one measurement per paper/catalyst/product for auditable ranking."""
    data = records.copy()
    data["fe_pct"] = pd.to_numeric(data["fe_pct"], errors="coerce")
    data["current_density_mA_cm2"] = pd.to_numeric(
        data.get("current_density_mA_cm2"), errors="coerce"
    )
    data = data.dropna(subset=["fe_pct"])
    keys = ["doi", "catalyst_name", "product"]
    return data.loc[data.groupby(keys, dropna=False)["fe_pct"].idxmax()].copy()


def _cell_matches(value: str, requested: str) -> bool:
    if not requested or requested == "不限":
        return True
    text = str(value or "").lower()
    aliases = {
        "H-cell": ("h-cell", "h-type", "h type"),
        "Flow cell": ("flow", "gas diffusion", "gde"),
        "MEA": ("mea", "membrane electrode"),
    }
    return any(token in text for token in aliases.get(requested, (requested.lower(),)))


def _score_rows(
    rows: pd.DataFrame,
    requested_cell: str,
    min_fe: float,
    min_current: float,
    synthesis_preference: str,
) -> pd.DataFrame:
    """Rank evidence relevance; this is not a prospective performance model."""
    if rows.empty:
        return rows
    data = rows.copy()
    fe_scale = max(float(data["fe_pct"].max()), 1.0)
    data["design_relevance_score"] = 55.0 * data["fe_pct"] / fe_scale
    if min_fe > 0:
        data["design_relevance_score"] += 10.0 * (data["fe_pct"] >= min_fe)
    cell_ok = data["cell_setup"].map(lambda value: _cell_matches(value, requested_cell))
    data["design_relevance_score"] += 10.0 * cell_ok
    current = data["current_density_mA_cm2"]
    if min_current > 0:
        data["design_relevance_score"] += 10.0 * (current >= min_current).fillna(False)
        data["design_relevance_score"] -= 5.0 * current.isna()
    if synthesis_preference.strip():
        wanted = synthesis_preference.strip().lower()
        synth_ok = data["synthesis_method"].fillna("").str.lower().str.contains(
            re.escape(wanted), regex=True
        )
        data["design_relevance_score"] += 8.0 * synth_ok
    completeness = data[["voltage_V", "electrolyte", "doi"]].notna().sum(axis=1)
    data["design_relevance_score"] += 7.0 * completeness / 3.0
    return data.sort_values(["design_relevance_score", "fe_pct"], ascending=False)


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "未记录"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "未记录"


def _evidence_bundle(rows: pd.DataFrame, literature: Iterable[dict]) -> str:
    lines = ["## 结构化实验基线"]
    if rows.empty:
        lines.append("没有找到满足目标产物与材料约束的结构化记录。")
    for number, (_, row) in enumerate(rows.head(12).iterrows(), start=1):
        lines.append(
            f"[E{number}] {_clean(row.get('catalyst_name'))}；产物 {_clean(row.get('product'))}；"
            f"FE {_clean(row.get('fe_pct'))}%；电位 {_clean(row.get('voltage_V'))} V "
            f"vs. {_clean(row.get('voltage_reference'))}；电流密度 "
            f"{_clean(row.get('current_density_mA_cm2'))} mA cm⁻²；电解质 "
            f"{_clean(row.get('electrolyte'))}；电解槽 {_clean(row.get('cell_setup'))}；"
            f"合成 {_clean(row.get('synthesis_method'))}；DOI: {_clean(row.get('doi'))}。"
        )

    lines.append("\n## 相关全文证据")
    lit_count = 0
    for lit_count, entry in enumerate(literature, start=1):
        sections = entry.get("sections", {})
        excerpt = sections.get("abstract") or sections.get("conclusion") or entry.get("summary") or ""
        excerpt = re.sub(r"\s+", " ", str(excerpt)).strip()[:1200]
        title = _clean(entry.get("title") or entry.get("pdf_file"))
        lines.append(
            f"[L{lit_count}] {title}；DOI: {_clean(entry.get('doi'))}。\n"
            f"证据片段：{excerpt or '当前条目无可用摘要片段。'}"
        )
    if not lit_count:
        lines.append("没有检索到可用的全文证据片段。")
    return "\n".join(lines)


def _baseline_report(rows: pd.DataFrame, warnings: list[str]) -> str:
    lines = [
        "## 数据库基线（尚未调用模型）",
        "下列分数仅衡量记录与本次约束的相关程度，不是新催化剂性能预测。",
    ]
    if warnings:
        lines.extend([f"- ⚠️ {warning}" for warning in warnings])
    if rows.empty:
        lines.append("\n没有找到满足材料与产物硬约束的实验记录，不能可靠生成候选。")
        return "\n".join(lines)
    lines.extend([
        "\n|证据|催化剂|产物|FE%|电流密度|电解槽|DOI|",
        "|---|---|---:|---:|---:|---|---|",
    ])
    for number, (_, row) in enumerate(rows.head(8).iterrows(), start=1):
        lines.append(
            f"|E{number}|{_clean(row.get('catalyst_name'))}|{_clean(row.get('product'))}|"
            f"{_clean(row.get('fe_pct'))}|{_clean(row.get('current_density_mA_cm2'))}|"
            f"{_clean(row.get('cell_setup'))}|{_clean(row.get('doi'))}|"
        )
    return "\n".join(lines)


def run_catalyst_design(
    objective: str,
    target_product: str,
    required_elements: str,
    excluded_elements: str,
    cell_type: str,
    min_fe: float,
    min_current_density: float,
    synthesis_preference: str,
    client,
    model: str = DEFAULT_MODEL,
) -> tuple[str, str]:
    """Run one evidence retrieval → review → hypothesis generation cycle."""
    product = (target_product or "").strip()
    if not product or product == "请选择":
        return "请先选择目标产物。", ""

    required = _tokens(required_elements)
    excluded = _tokens(excluded_elements)
    records = _representative_rows(load_records())
    records = records[records["product"].astype(str).str.upper() == product.upper()]
    for element in required:
        records = records[
            records["catalyst_name"].map(lambda name: _contains_element(name, element))
        ]
    for element in excluded:
        records = records[
            ~records["catalyst_name"].map(lambda name: _contains_element(name, element))
        ]

    warnings = []
    cell_matches = records["cell_setup"].map(lambda value: _cell_matches(value, cell_type))
    if cell_type != "不限" and not records.empty and not cell_matches.any():
        warnings.append(f"没有明确标注为 {cell_type} 的记录，保留其他电解槽作为低等级参考。")
    if min_fe > 0 and not records.empty and not (records["fe_pct"] >= min_fe).any():
        warnings.append(f"数据库候选均未达到 FE ≥ {min_fe:g}% 的目标。")
    if min_current_density > 0 and not records.empty:
        known_current = records["current_density_mA_cm2"].dropna()
        if known_current.empty:
            warnings.append("候选记录均缺失电流密度，无法核验电流密度目标。")
        elif not (known_current >= min_current_density).any():
            warnings.append(
                f"有电流密度记录的候选均未达到 {min_current_density:g} mA cm⁻²。"
            )

    ranked = _score_rows(
        records, cell_type, min_fe, min_current_density, synthesis_preference
    )
    query = " ".join(
        filter(None, [objective, product, required_elements, cell_type, synthesis_preference])
    )
    intent = detect_query_intent(query)
    intent["products"] = [product]
    literature = retrieve_literature(query, intent, top_k=5)
    evidence = _evidence_bundle(ranked, literature)
    baseline = _baseline_report(ranked, warnings)
    if ranked.empty:
        return baseline, evidence

    constraints = f"""研究目标：{objective or '未补充自然语言目标'}
目标产物：{product}
必需元素：{', '.join(required) or '不限'}
禁用元素：{', '.join(excluded) or '无'}
目标电解槽：{cell_type}
最低目标 FE：{min_fe:g}%（这是目标，不是已有事实）
最低目标电流密度：{min_current_density:g} mA cm⁻²（这是目标，不是已有事实）
合成偏好：{synthesis_preference or '不限'}"""
    prompt = f"""请完成一次半自主催化剂设计评审。

### 用户约束
{constraints}

### 检索证据
{evidence}

### 数据质量提示
{chr(10).join('- ' + item for item in warnings) if warnings else '- 未触发额外警告，但缺失字段仍需逐项注明。'}

请严格按以下结构输出：
1. 任务理解与可行性边界
2. 三个候选：A 文献邻域保守方案、B 机理驱动方案、C 探索方案
   每个候选写组成/结构、科学依据、相对基线的预期方向、证据编号、不确定性和否决条件
3. 多目标排序表（科学依据、可合成性、条件可比性、风险；不得伪造预测 FE）
4. 首选方案的可执行实验流程（先给参数范围，不把文献未提供的参数写成确定值）
5. 对照组、重复次数、产物定量和 FE 闭合检查
6. XRD/XPS/SEM-TEM/必要原位表征与关键判据
7. 失败模式以及下一轮如何调整
8. 明确写出“需要研究人员批准后才能进入实验”的停止点
"""
    if client is None:
        return baseline + '\n\n> 请设置 DEEPSEEK_API_KEY 并重启应用以生成 AI 设计建议。', evidence
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DESIGN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=3500,
        )
        design = response.choices[0].message.content.strip()
        return design, evidence
    except Exception as exc:
        return (
            baseline
            + f"\n\n> 智能体生成阶段调用失败：{exc}。数据库检索结果仍可用于人工设计。",
            evidence,
        )
