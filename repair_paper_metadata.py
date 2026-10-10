"""Audit and repair PDF-to-DOI links in the literature knowledge base.

The legacy builder used catalyst-name substring matching as a fallback.  That
can attach many unrelated PDFs to the same structured record.  This utility
only accepts a DOI found in the paper itself (OCR Markdown first, embedded PDF
text second), then links structured records by exact normalized DOI.

Usage:
    python repair_paper_metadata.py
    python repair_paper_metadata.py --apply
"""

import argparse
import csv
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import fitz
except ImportError:
    fitz = None


DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
SUPPLEMENT_RE = re.compile(r"\.s\d{3,4}$", re.IGNORECASE)


def normalize_doi(value: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    value = str(value or '').strip().lower()
    value = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', value)
    value = value.rstrip('.,;:)]}>')
    value = re.sub(r'(?:/-)?/dcsupplemental$', '', value, flags=re.IGNORECASE)
    # ACS supporting-information identifiers refer to the parent article.
    value = SUPPLEMENT_RE.sub('', value)
    return value


def extract_pdf_prefix(path: Path, pages: int = 3) -> str:
    if fitz is None:
        return ''
    try:
        with fitz.open(path) as doc:
            return '\n'.join(page.get_text() for page in list(doc)[:pages])
    except (RuntimeError, ValueError):
        return ''


def choose_doi(text: str) -> Tuple[str, str]:
    """Return a conservative DOI candidate and confidence label."""
    if not text:
        return '', 'none'
    candidates = []
    for match in DOI_RE.finditer(text[:50000]):
        doi = normalize_doi(match.group(0))
        if not doi:
            continue
        nearby = text[max(0, match.start() - 45):match.start()].lower()
        score = 0
        if 'doi.org/' in nearby or 'doi:' in nearby or 'doi ' in nearby:
            score += 4
        if match.start() < 12000:
            score += 2
        elif match.start() < 25000:
            score += 1
        candidates.append((score, match.start(), doi))
    if not candidates:
        return '', 'none'
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, _, best_doi = candidates[0]
    # A DOI explicitly labelled on the first pages is high confidence.  An
    # unlabelled early DOI is retained as medium and reported for review.
    confidence = 'high' if best_score >= 6 else 'medium' if best_score >= 2 else 'low'
    return best_doi, confidence


def records_by_doi(path: Path) -> Dict[str, List[Dict]]:
    frame = pd.read_csv(path)
    result: Dict[str, List[Dict]] = {}
    fields = (
        'catalyst_name', 'catalyst_type', 'product', 'fe_pct',
        'voltage_V', 'electrolyte',
    )
    for _, row in frame.iterrows():
        doi = normalize_doi(row.get('doi', ''))
        if not doi:
            continue
        record = {field: row.get(field, '') for field in fields}
        result.setdefault(doi, []).append(record)
    return result


def audit_entries(
    entries: List[Dict], pdf_dir: Path, ocr_dir: Path
) -> Tuple[List[Dict], List[Dict]]:
    old_counts = Counter(normalize_doi(e.get('doi', '')) for e in entries)
    repaired = []
    report = []
    for entry in entries:
        item = dict(entry)
        pdf_file = str(item.get('pdf_file', ''))
        stem = Path(pdf_file).stem
        ocr_path = ocr_dir / f'{stem}.md'
        pdf_path = pdf_dir / pdf_file
        source = ''
        detected = ''
        confidence = 'none'
        if ocr_path.exists():
            detected, confidence = choose_doi(
                ocr_path.read_text(encoding='utf-8', errors='replace')
            )
            source = 'ocr' if detected else ''
        if not detected and pdf_path.exists():
            detected, confidence = choose_doi(extract_pdf_prefix(pdf_path))
            source = 'pdf_text' if detected else ''

        old_doi = normalize_doi(item.get('doi', ''))
        new_doi = detected if confidence in {'high', 'medium'} else ''
        # Standalone manually added entries have no local PDF; retain them.
        if not pdf_path.exists() and not ocr_path.exists():
            new_doi = old_doi
            confidence = 'manual'
            source = 'existing_non_pdf'
        # A unique legacy DOI is kept only when there is no contrary evidence.
        elif not new_doi and old_doi and old_counts[old_doi] == 1:
            new_doi = old_doi
            confidence = 'legacy_unique'
            source = 'existing'

        item['doi'] = new_doi
        item['doi_source'] = source
        item['doi_confidence'] = confidence
        repaired.append(item)
        report.append({
            'pdf_file': pdf_file,
            'old_doi': old_doi,
            'new_doi': new_doi,
            'changed': str(old_doi != new_doi),
            'source': source,
            'confidence': confidence,
            'has_ocr': str(ocr_path.exists()),
        })
    return repaired, report


def write_report(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', default='outputs/knowledge_base.json')
    parser.add_argument('--records', default='outputs/structured_records.csv')
    parser.add_argument('--pdf-dir', default='gold_corpus/pdfs')
    parser.add_argument('--ocr-dir', default='outputs/ocr_markdown')
    parser.add_argument('--report', default='outputs/paper_ingestion_audit.csv')
    parser.add_argument(
        '--missing-ocr-report', default='outputs/pdfs_missing_ocr.txt'
    )
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    kb_path = Path(args.kb)
    entries = json.loads(kb_path.read_text(encoding='utf-8'))
    repaired, report = audit_entries(
        entries, Path(args.pdf_dir), Path(args.ocr_dir)
    )
    write_report(Path(args.report), report)
    missing_ocr = sorted(
        row['pdf_file'] for row in report
        if row['pdf_file'] and row['has_ocr'] == 'False'
        and (Path(args.pdf_dir) / row['pdf_file']).exists()
    )
    missing_path = Path(args.missing_ocr_report)
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.write_text('\n'.join(missing_ocr) + '\n', encoding='utf-8')

    changed = sum(row['changed'] == 'True' for row in report)
    confidence = Counter(row['confidence'] for row in report)
    print(f'Entries: {len(report)}; DOI changes proposed: {changed}')
    print('Confidence:', dict(confidence))
    print(f'Report: {args.report}')
    print(f'PDFs missing OCR: {len(missing_ocr)} -> {missing_path}')

    if not args.apply:
        print('Dry run only. Re-run with --apply after reviewing the report.')
        return

    linked_records = records_by_doi(Path(args.records))
    for entry in repaired:
        entry['records'] = linked_records.get(normalize_doi(entry.get('doi', '')), [])

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup = kb_path.with_name(f'{kb_path.stem}.before-doi-repair-{stamp}.json')
    shutil.copy2(kb_path, backup)
    kb_path.write_text(
        json.dumps(repaired, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    print(f'Backup: {backup}')
    print(f'Updated: {kb_path}')


if __name__ == '__main__':
    main()
