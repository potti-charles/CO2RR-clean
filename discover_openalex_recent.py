"""Supplement Crossref discovery with OpenAlex and merge by DOI."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import time
from typing import Dict, Iterable, List
from urllib.parse import urlencode

import pandas as pd

from discover_recent_papers import clean_markup, relevance, request_json, write_csv
from repair_paper_metadata import normalize_doi


OPENALEX_API = 'https://api.openalex.org/works'
QUERIES = (
    'copper CO2 electroreduction',
    'copper electrochemical carbon dioxide reduction',
    'Cu2O CuO CuOx CO2 reduction',
    'oxide-derived copper CO2 reduction',
    'copper alloy CO2 electroreduction',
    'CuSn CuZn CuAg CuBi CO2 electroreduction',
    'copper single atom MOF CO2 reduction',
    'tandem copper CO2 electroreduction',
    'copper gas diffusion electrode flow cell CO2 reduction',
    'copper CO2 reduction ethylene ethanol methane multicarbon',
)


def abstract_from_index(index: object) -> str:
    if not isinstance(index, dict):
        return ''
    positioned = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, str(word)))
    return ' '.join(word for _, word in sorted(positioned))


def iter_openalex(query: str, start: str, end: str, limit: int) -> Iterable[Dict[str, object]]:
    cursor = '*'
    fetched = 0
    while fetched < limit:
        params = {
            'search': query,
            'filter': f'from_publication_date:{start},to_publication_date:{end},type:article',
            'select': (
                'id,doi,title,publication_year,publication_date,primary_location,'
                'best_oa_location,open_access,abstract_inverted_index,type'
            ),
            'per-page': min(200, limit - fetched),
            'cursor': cursor,
        }
        payload = request_json(OPENALEX_API + '?' + urlencode(params))
        items = payload.get('results', [])
        if not isinstance(items, list) or not items:
            break
        yield from items
        fetched += len(items)
        meta = payload.get('meta', {})
        next_cursor = meta.get('next_cursor', '') if isinstance(meta, dict) else ''
        if not next_cursor or next_cursor == cursor:
            break
        cursor = str(next_cursor)
        time.sleep(0.12)


def location_fields(item: Dict[str, object]) -> Dict[str, object]:
    primary = item.get('primary_location') if isinstance(item.get('primary_location'), dict) else {}
    best_oa = item.get('best_oa_location') if isinstance(item.get('best_oa_location'), dict) else {}
    source = primary.get('source') if isinstance(primary.get('source'), dict) else {}
    pdf_urls = []
    for location in (best_oa, primary):
        url = str(location.get('pdf_url') or '')
        if url and url not in pdf_urls:
            pdf_urls.append(url)
    return {
        'journal': clean_markup(str(source.get('display_name') or '')),
        'publisher': clean_markup(str(source.get('host_organization_name') or '')),
        'landing_url': str(primary.get('landing_page_url') or item.get('doi') or item.get('id') or ''),
        'pdf_urls': json.dumps(pdf_urls, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default=str(date.today()))
    parser.add_argument('--max-per-query', type=int, default=1000)
    parser.add_argument(
        '--crossref-csv', type=Path,
        default=Path('outputs/recent_literature_2022_present_broad/all_candidates.csv'),
    )
    parser.add_argument(
        '--database', type=Path, default=Path('outputs/structured_records.csv')
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('outputs/recent_literature_2022_present_combined'),
    )
    args = parser.parse_args()

    known_frame = pd.read_csv(args.database, usecols=['doi'])
    known = {normalize_doi(value) for value in known_frame['doi'].dropna() if normalize_doi(value)}
    candidates: Dict[str, Dict[str, object]] = {}
    if args.crossref_csv.exists():
        for row in pd.read_csv(args.crossref_csv).fillna('').to_dict('records'):
            row['source_indexes'] = 'Crossref'
            candidates[str(row['doi'])] = row

    query_hits: Dict[str, int] = {}
    priority = {'include': 2, 'review': 1, 'exclude': 0}
    for query in QUERIES:
        count = 0
        for item in iter_openalex(query, args.start, args.end, args.max_per_query):
            doi = normalize_doi(item.get('doi'))
            if not doi:
                continue
            count += 1
            abstract = abstract_from_index(item.get('abstract_inverted_index'))
            pseudo_item = {'title': [item.get('title') or ''], 'abstract': abstract}
            decision, reason = relevance(pseudo_item)
            locations = location_fields(item)
            row = {
                'doi': doi,
                'title': clean_markup(str(item.get('title') or '')),
                'year': item.get('publication_year') or '',
                **locations,
                'abstract': clean_markup(abstract),
                'decision': decision,
                'reason': reason,
                'already_structured': doi in known,
                'pdf_status': 'not_attempted',
                'local_pdf': '',
                'matched_queries': json.dumps([query], ensure_ascii=False),
                'source_indexes': 'OpenAlex',
            }
            previous = candidates.get(doi)
            if previous is None:
                candidates[doi] = row
                continue
            sources = {part for part in str(previous.get('source_indexes', '')).split(';') if part}
            sources.add('OpenAlex')
            if priority[decision] > priority[str(previous.get('decision', 'exclude'))]:
                row['source_indexes'] = ';'.join(sorted(sources))
                candidates[doi] = row
            else:
                previous['source_indexes'] = ';'.join(sorted(sources))
                old_urls = json.loads(str(previous.get('pdf_urls') or '[]'))
                new_urls = json.loads(str(row.get('pdf_urls') or '[]'))
                previous['pdf_urls'] = json.dumps(
                    list(dict.fromkeys(old_urls + new_urls)), ensure_ascii=False
                )
        query_hits[query] = count

    rows = sorted(candidates.values(), key=lambda row: (int(row.get('year') or 0), str(row['doi'])))
    included = [row for row in rows if row['decision'] == 'include']
    new_rows = [row for row in included if not row['already_structured']]
    review_rows = [row for row in rows if row['decision'] == 'review']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'all_candidates.csv', rows)
    write_csv(args.output_dir / 'new_high_confidence.csv', new_rows)
    write_csv(args.output_dir / 'review_candidates.csv', review_rows)
    summary = {
        'generated_at': date.today().isoformat(),
        'date_range': [args.start, args.end],
        'openalex_queries': query_hits,
        'unique_candidates': len(rows),
        'high_confidence_total': len(included),
        'high_confidence_new': len(new_rows),
        'review_candidates': len(review_rows),
        'found_by_both_indexes': sum(row.get('source_indexes') == 'Crossref;OpenAlex' for row in rows),
        'openalex_only': sum(row.get('source_indexes') == 'OpenAlex' for row in rows),
        'with_pdf_url_new': sum(str(row.get('pdf_urls') or '') not in ('', '[]') for row in new_rows),
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
