# 示例数据

- `structured_records.csv`：少量带 DOI 的公开文献实验记录，保留应用字段。
- `knowledge_base.json`：与 CSV 对应的简短文献片段，标记 `sample_only=true`。
- `figure_index.json`：空图片索引；示例包不附论文图片或 PDF。
- `db_summary_for_recommendation.md`：示例范围说明。

首次启动将缺失的数据文件复制到 `outputs/`。已有同名文件不会被覆盖。
示例数值不是完整领域排名，不能用此小数据集训练可靠的性能预测模型。
