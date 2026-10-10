# CO2RR Catalyst Knowledge Base · 学生调试版

Gradio 数据库界面，支持结构化实验检索、文献问答、催化剂设计建议和实验反馈。
这个仓库只附带少量公开文献示例，不含完整数据库、原始文献库、模型权重或 manuscript。

## 安装与启动

建议 Python 3.10 或 3.11。进入项目目录后创建自己的虚拟环境：

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 使用：.venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
python scripts/start_app.py
```

浏览器打开终端显示的本地网址，通常为 `http://localhost:7860`。
未配置 API Key 也能启动、浏览示例数据和检查检索逻辑。
AI 问答、AI 设计和 AI 实验反馈分析需要学生自行配置密钥。
首次运行自动把 `data/sample/` 中缺失的输入复制到 `outputs/`，保留已有数据。

## 配置 API 与管理员密码

复制 `.env.example` 为 `.env`（Windows 可使用 `Copy-Item .env.example .env`），
仅在自己电脑的 `.env` 中填写 `DEEPSEEK_API_KEY`。
`python scripts/start_app.py` 自动读取这个文件，已设置的环境变量优先。
程序不会读取原机器上的私密密钥文件。

也可以通过环境变量启动：

```bash
export DEEPSEEK_API_KEY='填写自己的密钥'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
export CO2RR_ADMIN_PASSWORD='填写自己设置的管理员密码'
python app.py
```

Windows PowerShell 对应语法为 `$env:DEEPSEEK_API_KEY='填写自己的密钥'`。
默认模型为 `deepseek-chat`，可用环境变量 `DEEPSEEK_MODEL` 修改。
管理员密码只控制图片人工审核入口；留空时该入口不可用。
OCR 和图片分类需使用支持相应模型的服务，并设置脚本的模型参数及 BASE_URL。
`.env`、密钥文件和本地输出不要提交到 Git。

## 文件说明

| 文件 / 目录 | 用途 |
|---|---|
| `app.py` | Gradio Web 界面 |
| `co2rr_knowledge_query.py` | 数据库查询、TF-IDF 检索和文献问答 |
| `catalyst_design_agent.py` | 从检索证据生成设计建议 |
| `experiment_feedback.py` | 保存设计任务、校验实验反馈和迭代分析 |
| `runtime_config.py` | 项目路径、示例初始化和环境变量读取 |
| `config/` | 不含密钥的文献导入配置模板 |
| `docs/` | 文献导入和扩展工作流说明 |
| `scripts/` | 启动与示例准备脚本 |
| `data/sample/` | 小规模 CSV/JSON 示例 |
| `requirements-app.txt` | 启动应用所需依赖 |
| `requirements-tools.txt` | 可选 PDF、图谱、机器学习依赖 |
| `requirements-model.txt` | 可选本地 LoRA 推理依赖 |

其余核心脚本保留了原文件名，涵盖文献发现、PDF/OCR、元数据修复、
知识库构建、图片分类、关系提取、知识图谱及机器学习数据整理。
这些扩展脚本需要自己的输入文件，具体见 [扩展说明](docs/student_workflows.md)。
设计记录、实验反馈、检索索引等在本地 `outputs/` 中生成，不包含原作者的会话数据。

## 文献提取优先使用 LoRA

`ingest_new_papers.py` 默认优先加载自己的 Qwen2.5-7B + enriched_v7 LoRA，
模型权重独立存放，通过 `CO2RR_LORA_ADAPTER_PATH` 和 `CO2RR_LORA_BASE_MODEL` 指定。
安装 `requirements-model.txt` 后执行 `python ingest_new_papers.py --pdf-dir incoming_papers`。
本地加载或校验失败且配置了 API 密钥时才回退到 DeepSeek；
可在配置中设置 `allow_api_fallback=false` 关闭回退。
具体配置、输出格式限制和审核步骤见 [本地 LoRA 导入说明](docs/local_lora_ingestion.md)。
Web 问答、设计与反馈分析使用各自的 DeepSeek API 配置。

## 调试与验证

```bash
python scripts/prepare_sample_data.py
python -c "from co2rr_knowledge_query import load_records, build_tfidf_index; build_tfidf_index(); print(load_records().shape)"
python app.py --port 7861
```

可在界面浏览催化剂、产物和 FE 数据。没有 API Key 时，AI 问答会提示配置密钥。
示例中的 FE 排名仅针对示例子集，不能作为领域最高性能或模型训练结论。
使用真实科研数据前请核对 DOI 对应原文及实验条件。
