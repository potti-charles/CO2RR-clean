# 优先使用自己的 LoRA 模型提取文献

`ingest_new_papers.py` 默认配置为 `config/ingest_models.lora.example.json`：
先读取 PDF 文本，再调用本地 Qwen2.5-7B + `lora_checkpoints_json_enriched_v7`。
本地模型加载或提取校验失败时，只有配置了自己的 `DEEPSEEK_API_KEY` 才回退到 API。
将配置中的 `allow_api_fallback` 改为 `false` 可完全关闭云端回退。
`reviewer` 默认为 `null`，成功的本地提取不会另行调用云端审核。

## 模型与环境

模型权重仍放在原目录，不复制进学生版。需要单独安装本地模型依赖：

```bash
python -m pip install -r requirements-model.txt
export CO2RR_LORA_ADAPTER_PATH='/你的路径/lora_checkpoints_json_enriched_v7'
export CO2RR_LORA_BASE_MODEL='/你的路径/Qwen2.5-7B-Instruct'
```

当前服务器上可以直接指向原目录里的 adapter 和基础模型缓存。
`base_model` 留空时读取 adapter 的 `adapter_config.json` 里的基础模型路径；
该路径在本机不存在时需要设置 `CO2RR_LORA_BASE_MODEL`。
所有模型加载均为 `local_files_only=true`，程序不会自动下载大模型。
相对 adapter 路径按项目根目录解析。学生需使用自己机器上的路径，
不能只复制这份代码就认为已经拿到训练好的模型。

默认使用 float16；CPU 使用 float32。若 GPU 支持，可配置 `dtype=bfloat16`。
显存不足时可在安装 bitsandbytes 后设置 `load_in_4bit=true`（需要 CUDA）。
同一批处理的模型只加载一次。脚本按训练时的上下文长度将文献分块，
分块超过 `max_chunks` 时报告错误，不静默截断文献。

## 提取和入库

在项目根目录运行，先执行 dry-run：

```bash
python scripts/prepare_sample_data.py
python ingest_new_papers.py --pdf-dir incoming_papers
```

查看 `outputs/new_paper_ingestion/ingestion_audit.jsonl` 和 CSV 审计报告。
确认结果后再运行同一命令并添加 `--apply`。
这些 CLI 读取环境变量，不自动读取 `.env`；`.env` 自动加载仅用于 Web 启动脚本。

日志包含实际使用的 `primary_model`、`primary_backend`、`primary_usage`、
`fallback_used` 和本地失败原因。API 回退数据不会被标记成 LoRA 结果。
如只希望使用 API，可指定原模板：

```bash
python ingest_new_papers.py --pdf-dir incoming_papers --config config/ingest_models.example.json
```

## 输出格式与边界

现有 LoRA 训练输出是 `materials`、`products`、`efficiency` 三个字符串数组。
适配程序保留原训练提示词，仅将能在原文同一句中同时核对的催化剂、产物、
FE 和已提取电位转换为数据库记录；不会按数组下标强行对应产物和 FE。
多催化剂、多个不同 FE、无法明确绑定的情况进入 API 回退或人工审核。
未提取的电解质、电流密度、合成方法等字段保留为空或 unknown；
论文标题暂用 PDF 文件名并记录 `title_source=pdf_filename`，请人工核对。
有多个合格记录时按现有导入规则将最高 FE 标为代表记录，需要核对是否符合论文主产物。

本地模型成功运行不代表记录已通过科研质量审核。正式入库前应核对原文，
尤其是表格、对照实验、引文数据、合成条件和多产物分布。
扫描版 PDF 仍需先准备 OCR 文本；本地 LoRA 只处理文本。
Web 问答、设计与反馈分析继续使用单独配置的 DeepSeek API。
