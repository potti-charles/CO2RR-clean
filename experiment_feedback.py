"""Persistent experimental feedback loop for catalyst design sessions."""

import csv
import json
import math
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

from co2rr_knowledge_query import DEFAULT_MODEL
from runtime_config import OUTPUT_DIR


DESIGN_SESSIONS_PATH = OUTPUT_DIR / "design_sessions.json"
FEEDBACK_CSV_PATH = OUTPUT_DIR / "experiment_feedback.csv"
EXPERIMENT_ROOT = OUTPUT_DIR / "design_experiments"
_STORE_LOCK = threading.Lock()

FEEDBACK_FIELDS = [
    "design_id", "experiment_id", "round_number", "created_at", "sample_id",
    "catalyst", "target_product", "cell_setup", "potential_V",
    "voltage_reference", "electrolyte", "total_current_density_mA_cm2",
    "target_fe_pct", "other_products_fe_pct", "total_fe_pct",
    "partial_current_density_mA_cm2", "control_fe_pct",
    "control_current_density_mA_cm2", "stability_hours", "replicates",
    "synthesis_success", "observations", "validation_status",
    "attachment_count",
]

FEEDBACK_SYSTEM_PROMPT = """你是 CO2RR 实验反馈分析智能体。你只能依据提供的设计任务和 [X#] 实验事实分析。

规则：
1. 区分实验事实、程序计算值、推断和下一轮假设；不得编造缺失条件。
2. 不能把一次实验当作可重复结论，少于3次重复必须标记为初步结果。
3. 总FE不闭合、数值越界或条件缺失时，先建议核查数据，不能判定催化剂失败。
4. FE与分电流密度必须同时分析；条件不一致时不得把差异归因于材料。
5. 用户上传附件只证明文件存在；当前未提供附件内容时，不得声称已经解读XRD/XPS/SEM。
6. 下一轮只改变少量可辨识变量，保留对照组，并给出可证伪判据。
7. 不得声称已经发现新催化剂；最终判断和下一轮实验必须由研究人员批准。
"""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_sessions_unlocked() -> dict:
    if not DESIGN_SESSIONS_PATH.exists():
        return {"sessions": []}
    try:
        data = json.loads(DESIGN_SESSIONS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"sessions": []}
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), list):
        return {"sessions": []}
    return data


def _save_sessions_unlocked(data: dict) -> None:
    DESIGN_SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESIGN_SESSIONS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(DESIGN_SESSIONS_PATH)


def create_design_session(constraints: dict, design: str, evidence: str) -> str:
    """Persist a generated design and return a stable task identifier."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    design_id = f"DESIGN-{stamp}-{uuid.uuid4().hex[:6].upper()}"
    session = {
        "design_id": design_id,
        "created_at": _now(),
        "status": "awaiting_experiment",
        "constraints": constraints,
        "design_output": design,
        "design_evidence": evidence,
        "experiments": [],
        "next_rounds": [],
    }
    with _STORE_LOCK:
        data = _load_sessions_unlocked()
        data["sessions"].append(session)
        _save_sessions_unlocked(data)
    return design_id


def recent_design_sessions(limit: int = 12) -> str:
    """Render recent persisted tasks so they can be reopened after a refresh."""
    with _STORE_LOCK:
        sessions = _load_sessions_unlocked()["sessions"]
    if not sessions:
        return "尚无已保存设计任务。"
    lines = ["|设计任务|目标产物|状态|实验数|创建时间|", "|---|---|---|---:|---|"]
    for session in reversed(sessions[-limit:]):
        constraints = session.get("constraints", {})
        lines.append(
            f"|`{session.get('design_id')}`|{constraints.get('target_product', '未记录')}|"
            f"{session.get('status', '未记录')}|{len(session.get('experiments', []))}|"
            f"{session.get('created_at', '未记录')}|"
        )
    return "\n".join(lines)


def _number(value, name: str, minimum=None, maximum=None, optional=False):
    if value is None or str(value).strip() == "":
        if optional:
            return None
        raise ValueError(f"{name}不能为空")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须是数字") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name}必须是有限数字")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name}不能小于{minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name}不能大于{maximum}")
    return number


def _find_session(data: dict, design_id: str) -> dict:
    for session in data.get("sessions", []):
        if session.get("design_id") == design_id:
            return session
    raise ValueError("没有找到该设计任务编号，请从‘最近设计任务’复制完整编号")


def _copy_attachments(paths, design_id: str, experiment_id: str) -> list[str]:
    saved = []
    destination = EXPERIMENT_ROOT / design_id / experiment_id
    for raw_path in paths or []:
        source = Path(str(raw_path))
        if not source.is_file():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        safe_name = Path(source.name).name
        target = destination / safe_name
        counter = 1
        while target.exists():
            target = destination / f"{source.stem}_{counter}{source.suffix}"
            counter += 1
        shutil.copy2(source, target)
        saved.append(str(target))
    return saved


def _validate_feedback(record: dict, constraints: dict) -> tuple[str, list[str], list[str]]:
    errors = []
    warnings = []
    total_fe = record["total_fe_pct"]
    if not 0 <= record["target_fe_pct"] <= 100:
        errors.append("目标产物FE超出0–100%。")
    if not 0 <= record["other_products_fe_pct"] <= 100:
        errors.append("其他产物FE合计超出0–100%。")
    if total_fe > 105:
        errors.append(f"总FE为{total_fe:.1f}%，超过105%，应核查校准、流量和电荷计算。")
    elif total_fe < 80:
        warnings.append(f"总FE仅为{total_fe:.1f}%，可能存在未定量产物或分析损失。")
    elif total_fe > 100:
        warnings.append(f"总FE为{total_fe:.1f}%，轻微超过100%，应检查实验误差。")
    if record["replicates"] < 3:
        warnings.append("独立重复少于3次，只能视为初步结果。")
    requested_cell = str(constraints.get("cell_type") or "不限")
    if requested_cell != "不限" and record["cell_setup"] != requested_cell:
        warnings.append(
            f"实际电解槽为{record['cell_setup']}，与设计目标{requested_cell}不同，不能直接判定材料效应。"
        )
    if not record["voltage_reference"].strip():
        warnings.append("未填写参比电极，电位不可与文献可靠比较。")
    if not record["electrolyte"].strip():
        warnings.append("未填写电解质，条件比较不完整。")
    if record["control_fe_pct"] is None:
        warnings.append("未填写对照组FE，无法归因性能变化。")
    if errors:
        return "invalid_data", errors, warnings
    if record["replicates"] < 3 or total_fe < 80 or total_fe > 100:
        return "preliminary", errors, warnings
    target_fe = float(constraints.get("min_fe") or 0)
    target_current = float(constraints.get("min_current_density") or 0)
    meets_fe = record["target_fe_pct"] >= target_fe
    meets_current = (
        target_current <= 0
        or record["total_current_density_mA_cm2"] >= target_current
    )
    better_than_control = (
        record["control_fe_pct"] is None
        or record["target_fe_pct"] > record["control_fe_pct"]
    )
    if meets_fe and meets_current and better_than_control:
        return "meets_targets_pending_human_review", errors, warnings
    return "partial_or_not_met", errors, warnings


def _append_feedback_csv(record: dict) -> None:
    FEEDBACK_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = FEEDBACK_CSV_PATH.exists()
    with FEEDBACK_CSV_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEEDBACK_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in FEEDBACK_FIELDS})


def submit_experiment_feedback(
    design_id: str,
    sample_id: str,
    catalyst: str,
    cell_setup: str,
    potential_v,
    voltage_reference: str,
    electrolyte: str,
    total_current_density,
    target_fe,
    other_products_fe,
    control_fe,
    control_current_density,
    stability_hours,
    replicates,
    synthesis_success: str,
    observations: str,
    attachments,
    client,
    model: str = DEFAULT_MODEL,
) -> tuple[str, str, str]:
    """Validate, persist and analyze one experimental feedback record."""
    design_id = str(design_id or "").strip()
    if not design_id:
        return "请先生成设计任务或填写已有设计任务编号。", "", recent_design_sessions()
    try:
        with _STORE_LOCK:
            data = _load_sessions_unlocked()
            session = _find_session(data, design_id)
            constraints = dict(session.get("constraints", {}))
            previous_experiments = list(session.get("experiments", []))

        target_fe_value = _number(target_fe, "目标产物FE", 0, 100)
        other_fe_value = _number(other_products_fe, "其他产物FE合计", 0, 100)
        total_current_value = _number(total_current_density, "总电流密度", 0)
        replicate_value = int(_number(replicates, "独立重复次数", 1))
        record = {
            "design_id": design_id,
            "experiment_id": f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:5].upper()}",
            "round_number": len(session.get("experiments", [])) + 1,
            "created_at": _now(),
            "sample_id": str(sample_id or "").strip(),
            "catalyst": str(catalyst or "").strip(),
            "target_product": constraints.get("target_product", "未记录"),
            "cell_setup": str(cell_setup or "").strip(),
            "potential_V": _number(potential_v, "工作电位", optional=True),
            "voltage_reference": str(voltage_reference or "").strip(),
            "electrolyte": str(electrolyte or "").strip(),
            "total_current_density_mA_cm2": total_current_value,
            "target_fe_pct": target_fe_value,
            "other_products_fe_pct": other_fe_value,
            "total_fe_pct": target_fe_value + other_fe_value,
            "partial_current_density_mA_cm2": total_current_value * target_fe_value / 100.0,
            "control_fe_pct": _number(control_fe, "对照组FE", 0, 100, optional=True),
            "control_current_density_mA_cm2": _number(
                control_current_density, "对照组电流密度", 0, optional=True
            ),
            "stability_hours": _number(stability_hours, "稳定性时长", 0, optional=True),
            "replicates": replicate_value,
            "synthesis_success": str(synthesis_success or "").strip(),
            "observations": str(observations or "").strip(),
        }
        status, errors, warnings = _validate_feedback(record, constraints)
        record["validation_status"] = status
        saved_files = _copy_attachments(
            attachments, design_id, record["experiment_id"]
        )
        record["attachments"] = saved_files
        record["attachment_count"] = len(saved_files)

        x_evidence = "\n".join([
            f"[X1] 样品 {record['sample_id'] or '未编号'}；实际催化剂：{record['catalyst'] or '未填写'}；"
            f"合成是否成功：{record['synthesis_success'] or '未填写'}。",
            f"[X2] 电解槽 {record['cell_setup'] or '未填写'}；电位 {record['potential_V']} V vs. "
            f"{record['voltage_reference'] or '未填写'}；电解质 {record['electrolyte'] or '未填写'}。",
            f"[X3] 目标产物 {record['target_product']} FE={target_fe_value:.2f}%；其他产物FE合计="
            f"{other_fe_value:.2f}%；程序计算总FE={record['total_fe_pct']:.2f}%。",
            f"[X4] 总电流密度={total_current_value:.2f} mA cm⁻²；程序计算目标产物分电流密度="
            f"{record['partial_current_density_mA_cm2']:.2f} mA cm⁻²。",
            f"[X5] 对照组FE={record['control_fe_pct']}%；对照组总电流密度="
            f"{record['control_current_density_mA_cm2']} mA cm⁻²；独立重复={replicate_value}次；"
            f"稳定性={record['stability_hours']} h。",
            f"[X6] 实验观察：{record['observations'] or '未填写'}。附件数量={len(saved_files)}；"
            "附件尚未在本步骤中进行视觉或原始数据解析。",
        ])
        validation = [f"程序状态：`{status}`"]
        validation.extend(f"错误：{item}" for item in errors)
        validation.extend(f"警告：{item}" for item in warnings)
        audit = "\n".join(validation) + "\n\n" + x_evidence

        if errors:
            analysis = (
                "## 数据尚未通过校验\n\n"
                + "\n".join(f"- {item}" for item in errors)
                + "\n\n记录已保存用于追溯，但不会据此生成下一轮材料结论。请修正后重新提交。"
            )
            next_round = ""
        elif client is None:
            analysis = (
                "## 程序校验已完成\n\n"
                + audit
                + "\n\n请设置 DEEPSEEK_API_KEY 并重启应用以生成 AI 实验反馈分析。"
            )
            next_round = ""
        else:
            history_lines = []
            for previous in previous_experiments:
                history_lines.append(
                    f"- {previous.get('experiment_id')}：样品 {previous.get('sample_id') or '未编号'}；"
                    f"目标产物FE={previous.get('target_fe_pct')}%；总电流密度="
                    f"{previous.get('total_current_density_mA_cm2')} mA cm⁻²；分电流密度="
                    f"{previous.get('partial_current_density_mA_cm2')} mA cm⁻²；"
                    f"状态={previous.get('validation_status')}；观察={previous.get('observations') or '未填写'}。"
                )
            history_context = "\n".join(history_lines) or "无，这是第一次实验反馈。"
            prompt = f"""### 原设计任务
任务编号：{design_id}
设计目标与约束：{json.dumps(constraints, ensure_ascii=False)}
原设计方案：{session.get('design_output', '')[:7000]}

### 原设计的数据库与文献证据
{session.get('design_evidence', '')[:10000]}

### 此前实验历史
{history_context}

### 本轮真实实验事实
{x_evidence}

### 程序校验
{chr(10).join(validation)}

请输出：
1. 本轮是成功、部分成功、未达目标还是数据不足，并说明判定边界
2. FE与目标产物分电流密度相对于目标和对照的分析
3. 哪些现象可以由数据支持，哪些仅是假设
4. 下一轮只改变1–2个关键变量的实验方案和完整对照
5. 需要补充的表征或定量数据，以及明确的否决条件
6. 最后写明“等待研究人员批准下一轮实验”
"""
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.15,
                    max_tokens=3000,
                )
                analysis = response.choices[0].message.content.strip()
            except Exception as exc:
                analysis = (
                    "## 程序校验已完成，但模型分析失败\n\n"
                    f"{audit}\n\nAPI错误：{exc}"
                )
            next_round = f"ROUND-{len(session.get('next_rounds', [])) + 2:02d}"

        record["audit"] = audit
        record["agent_analysis"] = analysis
        record["proposed_next_round"] = next_round
        with _STORE_LOCK:
            data = _load_sessions_unlocked()
            session = _find_session(data, design_id)
            session.setdefault("experiments", []).append(record)
            if next_round:
                session.setdefault("next_rounds", []).append({
                    "round_id": next_round,
                    "based_on": record["experiment_id"],
                    "created_at": _now(),
                    "proposal": analysis,
                    "status": "awaiting_human_approval",
                })
                session["status"] = "next_round_awaiting_approval"
            elif client is None and not errors:
                session["status"] = "feedback_saved_pending_analysis"
            else:
                session["status"] = "feedback_needs_correction"
            _append_feedback_csv(record)
            _save_sessions_unlocked(data)
        heading = (
            f"## 实验反馈已保存\n\n- 任务：`{design_id}`\n"
            f"- 实验：`{record['experiment_id']}`\n- 数据状态：`{status}`\n"
            f"- 下一轮：`{next_round or '暂不生成'}`\n\n"
        )
        return heading + analysis, audit, recent_design_sessions()
    except ValueError as exc:
        return f"无法提交：{exc}", "", recent_design_sessions()
