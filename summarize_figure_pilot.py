"""Summarize the scientific-figure vision classification pilot."""

from collections import Counter, defaultdict
import json
from pathlib import Path


INPUT = Path('outputs/figure_reclassification_pilot.json')
OUTPUT = Path('outputs/figure_reclassification_pilot_report.md')


def main() -> None:
    rows = json.loads(INPUT.read_text(encoding='utf-8'))
    status = Counter(row['status'] for row in rows)
    transitions = Counter(
        (row['old_category'], row['new_category']) for row in rows
    )
    confidence_bins = Counter()
    for row in rows:
        confidence = float(row.get('confidence', 0))
        if confidence >= 0.90:
            confidence_bins['0.90–1.00'] += 1
        elif confidence >= 0.85:
            confidence_bins['0.85–0.90'] += 1
        elif confidence >= 0.70:
            confidence_bins['0.70–0.85'] += 1
        else:
            confidence_bins['<0.70'] += 1

    lines = [
        '# Scientific Figure Reclassification Pilot', '',
        f'- Total: {len(rows)}',
        f"- Accepted: {status['accepted']}",
        f"- Manual review: {status['manual_review']}",
        f"- API errors: {status['api_error']}",
        f"- Changed category: {sum(r['old_category'] != r['new_category'] for r in rows)}",
        f"- Caption conflicts: {sum(bool(r.get('caption_conflict')) for r in rows)}",
        f"- Second-model reviews: {sum(bool(r.get('review')) for r in rows)}",
        '', '## Confidence distribution', '',
    ]
    for label in ('0.90–1.00', '0.85–0.90', '0.70–0.85', '<0.70'):
        lines.append(f'- {label}: {confidence_bins[label]}')

    lines.extend(['', '## Category transitions', '', '| Old | New | Count |', '|---|---|---:|'])
    for (old, new), count in transitions.most_common():
        lines.append(f'| {old} | {new} | {count} |')

    lines.extend(['', '## Manual review / failures', ''])
    flagged = [row for row in rows if row['status'] != 'accepted']
    if not flagged:
        lines.append('None.')
    for row in flagged:
        lines.append(
            f"- `{row['file']}`: {row['old_category']} → {row['new_category']}; "
            f"confidence={row['confidence']}; status={row['status']}"
        )

    lines.extend(['', '## Lowest-confidence accepted results', ''])
    accepted = sorted(
        (row for row in rows if row['status'] == 'accepted'),
        key=lambda row: row['confidence'],
    )[:15]
    for row in accepted:
        lines.append(
            f"- `{row['file']}`: {row['old_category']} → {row['new_category']}; "
            f"confidence={row['confidence']}"
        )

    OUTPUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {OUTPUT}')


if __name__ == '__main__':
    main()
