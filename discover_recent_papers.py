"""Discover recent Cu-based electrochemical CO2-reduction papers.

Crossref is used as the reproducible, no-key discovery source.  Results are
deduplicated against the structured database and written to an audit bundle.
Publisher PDF links exposed in Crossref metadata are downloaded only when the
response is an actual PDF; every unresolved item remains in a manual manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import csv
from datetime import date
import json
from pathlib import Path
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from repair_paper_metadata import normalize_doi


CROSSREF_API = 'https://api.crossref.org/works'
QUERIES = (
    'copper electrochemical carbon dioxide reduction',
    'copper CO2 electroreduction',
    'Cu CO2 electrochemical reduction Faradaic efficiency',
    'copper carbon dioxide reduction electrocatalyst',
    'Cu-based CO2RR catalyst',
    'copper CO2 reduction ethylene ethanol formate carbon monoxide',
    'copper CO2 reduction methane ethylene ethanol propanol acetate',
    'Cu2O CO2 reduction',
    'CuO CO2 reduction',
    'CuOx CO2 reduction',
    'oxide-derived copper CO2 reduction',
    'copper alloy CO2 electroreduction',
    'CuSn CO2 electroreduction',
    'CuZn CO2 electroreduction',
    'CuAg CO2 electroreduction',
    'CuBi CO2 electroreduction',
    'copper single atom CO2 reduction',
    'copper metal organic framework CO2 reduction',
    'copper molecular catalyst CO2 reduction',
    'tandem copper CO2 electroreduction',
    'copper gas diffusion electrode CO2 reduction',
    'copper flow cell CO2 reduction',
    'copper carbon dioxide reduction C2+',
    'copper carbon dioxide reduction multicarbon',
)
CO2_RE = re.compile(r'(?<![A-Za-z])(?:co\s*2|co₂|carbon dioxide)(?![A-Za-z])', re.I)
ELECTRO_RE = re.compile(
    r'electro(?:chemical|catal|reduct)|co2rr|faradai|electrolysis|cathod', re.I
)
REDUCTION_RE = re.compile(r'reduct|conversion|co2rr|electrolysis', re.I)
CU_RE = re.compile(
    r'\b(?:copper|cuprous|cupric)\b|(?<![A-Za-z])(?:Cu|cu)(?=$|[^a-z])'
)
REVIEW_RE = re.compile(
    r'\b(?:review|perspective|progress|advances|roadmap|overview)\b', re.I
)
NON_ELECTRO_RE = re.compile(
    r'\b(?:photocatal|photoelectro|thermocatal|hydrogenation|dry reforming|'
    r'battery|batteries|cycloaddition)\b', re.I
)


def clean_markup(value: str) -> str:
    value = re.sub(r'<[^>]+>', ' ', str(value or ''))
    return re.sub(r'\s+', ' ', value).strip()


def first(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value or '')


def publication_year(item: Dict[str, object]) -> Optional[int]:
    for key in ('published-print', 'published-online', 'published', 'issued'):
        parts = item.get(key)
        if isinstance(parts, dict):
            date_parts = parts.get('date-parts')
            if isinstance(date_parts, list) and date_parts and date_parts[0]:
                try:
                    return int(date_parts[0][0])
                except (TypeError, ValueError):
                    pass
    return None


def request_json(url: str, retries: int = 4) -> Dict[str, object]:
    headers = {'User-Agent': 'CO2RR-curation/1.0 (mailto:local-curator@example.org)'}
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                return json.loads(response.read().decode('utf-8'))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f'Crossref request failed: {last_error}')


def iter_crossref(query: str, start: str, end: str, rows: int) -> Iterable[Dict[str, object]]:
    """Page with offsets; this avoids cursor repetition seen for relevance queries."""
    offset = 0
    while offset < rows:
        batch_size = min(200, rows - offset)
        params = {
            'query.bibliographic': query,
            'filter': f'from-pub-date:{start},until-pub-date:{end},type:journal-article',
            'rows': batch_size,
            'offset': offset,
        }
        payload = request_json(CROSSREF_API + '?' + urlencode(params))
        message = payload.get('message', {})
        items = message.get('items', []) if isinstance(message, dict) else []
        if not items:
            break
        yield from items
        offset += len(items)
        if len(items) < batch_size:
            break
        time.sleep(0.15)


def relevance(item: Dict[str, object]) -> Tuple[str, str]:
    title = clean_markup(first(item.get('title')))
    abstract = clean_markup(str(item.get('abstract') or ''))
    text = f'{title} {abstract}'
    has_co2 = bool(CO2_RE.search(text))
    has_reduction = bool(REDUCTION_RE.search(text))
    has_electro = bool(ELECTRO_RE.search(text))
    has_cu = bool(CU_RE.search(text))
    if has_co2 and has_reduction and has_cu:
        if REVIEW_RE.search(title):
            return 'review', 'cu_co2rr_review'
        if NON_ELECTRO_RE.search(title) and not has_electro:
            return 'exclude', 'non_electrochemical_route'
        if has_electro:
            return 'include', 'cu_co2_electroreduction'
        return 'review', 'possible_cu_co2_reduction'
    if has_co2 and has_electro and not has_cu:
        return 'review', 'cu_not_explicit_in_metadata'
    return 'exclude', 'topic_terms_incomplete'


def pdf_links(item: Dict[str, object]) -> List[str]:
    links = []
    for entry in item.get('link', []) if isinstance(item.get('link'), list) else []:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get('URL') or '')
        content_type = str(entry.get('content-type') or '').lower()
        if url and ('pdf' in content_type or url.lower().endswith('.pdf')):
            links.append(url)
    return list(dict.fromkeys(links))


def safe_name(doi: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', doi).strip('_') + '.pdf'


def download_pdf(urls: List[str], target: Path) -> Tuple[bool, str]:
    headers = {'User-Agent': 'Mozilla/5.0 CO2RR-curation/1.0'}
    for url in urls:
        try:
            with urlopen(Request(url, headers=headers), timeout=20) as response:
                data = response.read()
            if data[:5] == b'%PDF-' and len(data) > 5000:
                target.write_bytes(data)
                return True, url
        except (HTTPError, URLError, TimeoutError, OSError):
            continue
    return False, ''


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else [
        'doi', 'title', 'year', 'journal', 'publisher', 'landing_url',
        'abstract', 'decision', 'reason', 'already_structured', 'pdf_urls',
        'pdf_status', 'local_pdf',
    ]
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default=str(date.today()))
    parser.add_argument('--max-per-query', type=int, default=1000)
    parser.add_argument('--database', type=Path, default=Path('outputs/structured_records.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/recent_literature_2022_present'))
    parser.add_argument('--download-pdfs', action='store_true')
    parser.add_argument('--download-workers', type=int, default=8)
    args = parser.parse_args()

    known = set()
    if args.database.exists():
        frame = pd.read_csv(args.database, usecols=['doi'])
        known = {normalize_doi(value) for value in frame['doi'].dropna() if normalize_doi(value)}

    candidates: Dict[str, Dict[str, object]] = {}
    query_hits: Dict[str, int] = {}
    for query in QUERIES:
        count = 0
        for item in iter_crossref(query, args.start, args.end, args.max_per_query):
            doi = normalize_doi(item.get('DOI'))
            if not doi:
                continue
            count += 1
            decision, reason = relevance(item)
            row = {
                'doi': doi,
                'title': clean_markup(first(item.get('title'))),
                'year': publication_year(item) or '',
                'journal': clean_markup(first(item.get('container-title'))),
                'publisher': clean_markup(str(item.get('publisher') or '')),
                'landing_url': str(item.get('URL') or f'https://doi.org/{doi}'),
                'abstract': clean_markup(str(item.get('abstract') or '')),
                'decision': decision,
                'reason': reason,
                'already_structured': doi in known,
                'pdf_urls': json.dumps(pdf_links(item), ensure_ascii=False),
                'pdf_status': 'not_attempted',
                'local_pdf': '',
                'matched_queries': json.dumps([query], ensure_ascii=False),
            }
            previous = candidates.get(doi)
            priority = {'include': 2, 'review': 1, 'exclude': 0}
            if previous is None or priority[decision] > priority[str(previous['decision'])]:
                if previous is not None:
                    old_queries = json.loads(str(previous['matched_queries']))
                    row['matched_queries'] = json.dumps(
                        list(dict.fromkeys(old_queries + [query])), ensure_ascii=False
                    )
                candidates[doi] = row
            elif previous is not None:
                old_queries = json.loads(str(previous['matched_queries']))
                previous['matched_queries'] = json.dumps(
                    list(dict.fromkeys(old_queries + [query])), ensure_ascii=False
                )
        query_hits[query] = count

    rows = sorted(candidates.values(), key=lambda row: (int(row['year'] or 0), str(row['doi'])))
    new_rows = [row for row in rows if row['decision'] == 'include' and not row['already_structured']]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = args.output_dir / 'pdfs'
    pdf_dir.mkdir(exist_ok=True)
    if args.download_pdfs:
        jobs = {}
        for row in new_rows:
            urls = json.loads(str(row['pdf_urls']))
            if not urls:
                row['pdf_status'] = 'no_crossref_pdf_url'
                continue
            target = pdf_dir / safe_name(str(row['doi']))
            if target.exists() and target.read_bytes()[:5] == b'%PDF-':
                row['pdf_status'] = 'already_downloaded'
                row['local_pdf'] = str(target)
                continue
            jobs[(tuple(urls), target)] = row
        with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as executor:
            futures = {
                executor.submit(download_pdf, list(urls), target): row
                for (urls, target), row in jobs.items()
            }
            for index, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                ok, source = future.result()
                target = pdf_dir / safe_name(str(row['doi']))
                row['pdf_status'] = 'downloaded' if ok else 'download_failed'
                row['local_pdf'] = str(target) if ok else ''
                if ok:
                    row['pdf_urls'] = json.dumps([source], ensure_ascii=False)
                if index % 25 == 0:
                    print(f'PDF progress: {index}/{len(futures)}', flush=True)

    unresolved = [row for row in new_rows if not row['local_pdf']]
    write_csv(args.output_dir / 'all_candidates.csv', rows)
    write_csv(args.output_dir / 'new_relevant.csv', new_rows)
    write_csv(args.output_dir / 'missing_pdfs.csv', unresolved)
    summary = {
        'generated_at': date.today().isoformat(),
        'date_range': [args.start, args.end],
        'queries': query_hits,
        'unique_candidates': len(rows),
        'included_total': sum(row['decision'] == 'include' for row in rows),
        'already_structured': sum(row['decision'] == 'include' and row['already_structured'] for row in rows),
        'new_relevant': len(new_rows),
        'pdf_downloaded': sum(bool(row['local_pdf']) for row in new_rows),
        'missing_pdfs': len(unresolved),
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
