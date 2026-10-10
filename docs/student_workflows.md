# 扩展工作流

最小应用只需 `requirements-app.txt`，不依赖 CUDA 或本地大模型。
以下工具属于可选扩展，先运行 `python -m pip install -r requirements-tools.txt`。
大多数旧脚本保留相对路径约定，请在项目根目录执行，并自行准备输入。

| 脚本 | 所需输入 |
|---|---|
| `ingest_new_papers.py` | 自己的 PDF 目录，默认使用 `config/ingest_models.lora.example.json` 和自己的 LoRA / 基础模型路径；可选 API 回退 |
| `repair_paper_metadata.py` | 本地知识库、结构化记录及自己的 PDF/可选 OCR |
| `discover_recent_papers.py` / `discover_openalex_recent.py` | 当前结构化记录和网络连接 |
| `build_knowledge_base.py` | 自己的 PDF 目录及结构化记录 |
| `ocr_pdf_to_markdown.py` | 自己的 PDF 和支持 OCR 的模型服务 |
| `extract_and_classify_figures.py` / `reclassify_figures_*.py` | 自己的 PDF/图片、图片索引和支持视觉模型的服务 |
| `extract_synthesis_methods.py` | 自己的 PDF 文本 / OCR 及结构化记录 |
| `normalize_kg.py` | `relation_extraction_results/relation_results.jsonl` |
| `build_knowledge_graph.py` / `visualize_kg.py` | `relation_extraction_results/normalized_relations.csv` |
| `build_ml_dataset.py` / `train_xgboost.py` | 自己的催化剂排行榜 / ML CSV |
| `build_ml_dataset_v2.py` / `train_fe_predictor.py` | 自己的 gold corpus Excel / ML CSV |

`build_ml_dataset_v2.py` 默认读取 `data/fe_record_corrections.csv`；没有时跳过修正。
该修正文件未包含在学生版。`summarize_figure_pilot.py` 需先生成图片审核结果。

运行 `python <脚本名> --help` 查看支持的参数；无 CLI 的旧脚本需检查顶部路径配置。
原始完整数据、原图、模型权重及研究论文写作文件均不在此仓库中。

## 本地模型扩展

文献导入已接入本地 LoRA 优先流程，见 [LoRA 导入说明](local_lora_ingestion.md)。

`extract_relations.py` 和 `inference_lora_json.py` 是可选本地模型工具，
需要 `requirements-model.txt`、学生自行下载的基础模型及自己的 LoRA adapter。
使用脚本参数设置模型路径；模型缓存默认使用本机 Hugging Face 缓存，
可以通过 `HF_HUB_CACHE` 环境变量调整。大模型推理通常需要足够内存 / GPU。
不运行这些工具时无需安装这组依赖。

## 导入自己的文献

见 [批量导入说明](batch_paper_ingestion.md)。先准备示例数据，再运行默认 dry-run，
审核输出后使用 `--apply`。更新后重启应用以重建内存中的检索状态；
若知识库内容改变，请删除自己本地的 `outputs/tfidf_index.pkl` 让程序重建索引。
