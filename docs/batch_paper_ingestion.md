# Batch ingestion of new CO2RR papers

The student edition defaults to local LoRA extraction with optional API fallback.
See [local LoRA setup](local_lora_ingestion.md). The commands below explicitly
select the API-only configuration and remain available as an alternative.

`ingest_new_papers.py` adds new PDF papers through an evidence-grounded,
reviewable workflow. It replaces the legacy approach of independently merging
material, product and FE strings.

## Safety model

- Dry-run is the default; the database is unchanged without `--apply`.
- DOI is read from the paper and checked against the existing database.
- Every FE is stored together with its product, matching potential, reference
  electrode and a short source excerpt.
- Exactly one product is marked `is_representative=true`; explicit secondary
  products remain available for analysis but are not shown as headline results.
- FE must be in `(0, 100]` and both FE and potential must occur in the evidence.
- A second model can accept, correct or reject the primary extraction.
- Conflicting existing DOI/catalyst/product keys are sent to review, never
  silently overwritten.
- `--apply` creates timestamped backups before changing CSV or JSON data.

## Model configuration

Copy the example without putting API keys in the file:

```bash
cp config/ingest_models.example.json config/ingest_models.json
export CO2RR_PRIMARY_API_KEY='...'
export CO2RR_REVIEW_API_KEY='...'
```

Edit `base_url` and `model` for the selected providers. Any service implementing
OpenAI-compatible `POST /v1/chat/completions` can be used. The primary and
reviewer should preferably be different model families to reduce correlated
errors. If a provider rejects `response_format`, set it to `false` for that
profile.

## Run

Place new PDFs in a separate directory, then run a dry-run:

```bash
.venv/bin/python ingest_new_papers.py \
  --pdf-dir incoming_papers \
  --config config/ingest_models.json
```

Review:

- `outputs/new_paper_ingestion/ingestion_summary.csv`
- `outputs/new_paper_ingestion/accepted_records.csv`
- `outputs/new_paper_ingestion/ingestion_audit.jsonl`

Statuses are `accepted`, `review`, `duplicate`, or `error`. After reviewing the
audit bundle, apply accepted records:

```bash
.venv/bin/python ingest_new_papers.py \
  --pdf-dir incoming_papers \
  --config config/ingest_models.json \
  --apply
```

The command updates `outputs/structured_records.csv` and
`outputs/knowledge_base.json`. Restart the web app after a successful apply so
its in-memory retrieval index sees the new papers.

## Scanned PDFs

When embedded PDF text is insufficient, the command automatically checks
`outputs/ocr_markdown/<pdf-stem>.md` (override with `--ocr-dir`). If neither
source has enough text, the paper receives `review` status with `OCR required`.

## Offline smoke test

`--fixture-response fixture.json` bypasses the API for one PDF and exercises
DOI detection, schema validation, audit generation and dry-run behavior. It is
intended for development only.
