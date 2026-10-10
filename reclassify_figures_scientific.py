"""Vision-first scientific figure classification with calibrated evidence scores.

The script never overwrites ``figure_index.json`` unless ``--apply`` is used.
Its default pilot targets the historically noisy XRD/electrochemistry groups.
"""

import argparse
import base64
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


INDEX_PATH = Path('outputs/figure_index.json')
PILOT_JSON = Path('outputs/figure_reclassification_pilot.json')
PILOT_CSV = Path('outputs/figure_reclassification_pilot.csv')
PRIMARY_MODEL = 'ERNIE-4.5-Turbo-VL-32K'
REVIEW_MODEL = 'Kimi-K2.5'

CATEGORIES = [
    'xrd', 'electron_microscopy', 'xps', 'raman_ftir', 'xas',
    'electrochemistry', 'faradaic_efficiency', 'current_density',
    'stability', 'theoretical_calculation', 'thermal_analysis',
    'adsorption_analysis', 'schematic', 'other',
]

CATEGORY_FEATURES = {
    'xrd': ['2θ', '2-theta', 'diffraction', 'bragg', 'jcpds', 'crystal plane'],
    'electron_microscopy': ['sem', 'tem', 'stem', 'haadf', 'micrograph', 'scale bar', 'nm', 'μm'],
    'xps': ['binding energy', 'xps', '2p3/2', '2p1/2'],
    'raman_ftir': ['raman shift', 'wavenumber', 'd-band', 'g-band', 'ftir'],
    'xas': ['xanes', 'exafs', 'k-edge', 'r space', 'χ(k)'],
    'electrochemistry': ['potential', 'voltage', 'tafel', 'nyquist', 'impedance', 'cv', 'lsv'],
    'faradaic_efficiency': ['faradaic efficiency', 'faradaic', 'selectivity', 'fe (%)', 'fe%'],
    'current_density': ['current density', 'ma cm', 'a cm', 'partial current'],
    'stability': ['time (h', 'time/h', 'hours', 'stability', 'durability', 'chrono'],
    'theoretical_calculation': ['free energy', 'reaction coordinate', 'pdos', 'dos', 'dft', 'charge density', 'adsorption energy'],
    'thermal_analysis': ['weight (%)', 'temperature', 'tga', 'dsc', 'thermograv'],
    'adsorption_analysis': ['relative pressure', 'p/p0', 'pore diameter', 'bet', 'adsorption-desorption'],
    'schematic': ['schematic', 'workflow', 'reaction pathway', 'synthesis route'],
}

PROMPT = """Classify this CO2 electroreduction research figure by its VISUAL CONTENT.

Allowed categories:
- xrd: diffraction patterns, normally 2-theta versus intensity
- electron_microscopy: SEM/TEM/STEM/HAADF/EDS/SAED and particle-size histograms attached to microscopy
- xps: XPS spectra and fitted binding-energy peaks
- raman_ftir: Raman or FTIR spectra
- xas: XANES/EXAFS spectra
- electrochemistry: CV, LSV, Tafel, EIS/Nyquist or polarization curves; do not use for FE/current/stability if a more specific category applies
- faradaic_efficiency: FE/selectivity/product-distribution plots
- current_density: current-density performance plots
- stability: time-dependent durability/chronoamperometry
- theoretical_calculation: DFT free-energy diagrams, reaction coordinates, adsorption structures, charge density, DOS/PDOS or band structures
- thermal_analysis: TGA/DSC/weight-temperature plots
- adsorption_analysis: BET isotherms or pore-size distributions
- schematic: synthesis/reaction/apparatus diagrams
- other: none of the above

For a composite figure, choose the scientifically dominant category and list other present categories as secondary_categories. Read axes, scale bars and panel content. The supplied caption may be incorrectly paired, so use it only as weak auxiliary evidence.

Caption: {caption}

Return one compact valid JSON object only. Use plain strings without quotation marks inside the strings:
{{"primary_category":"...","secondary_categories":[],"confidence":0.0,
"visible_axes_text":"...","visual_evidence":"...","caption_consistent":true,
"is_composite":false}}"""

REVIEW_PROMPT = """Independently classify this scientific figure using visual content only.
Ignore any prior label and do not rely on a caption. Allowed categories: {categories}.
Return one compact valid JSON object only. visible_axes_text and visual_evidence
must each be a single plain string. Include primary_category,
secondary_categories, confidence, visible_axes_text, visual_evidence, and
is_composite."""


def get_client():
    from openai import OpenAI
    from runtime_config import api_config

    api_key, base_url = api_config()
    return OpenAI(api_key=api_key, base_url=base_url, timeout=150)


def image_path(figure: Dict) -> Path:
    explicit = Path(str(figure.get('output_path', '')))
    if explicit.exists():
        return explicit
    return Path('outputs/figures') / str(figure.get('category', 'other')) / figure['file']


def parse_json(text: str) -> Dict:
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    start, end = cleaned.find('{'), cleaned.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('No JSON object in model response')
    return json.loads(cleaned[start:end + 1])


def normalized_confidence(value) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return {'high': 0.90, 'medium': 0.70, 'low': 0.45}.get(str(value).lower(), 0.60)


def call_model(client: OpenAI, model: str, path: Path, prompt: str) -> Dict:
    encoded = base64.b64encode(path.read_bytes()).decode()
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}'}},
                ]}],
                temperature=0,
                max_tokens=500,
            )
            result = parse_json(response.choices[0].message.content)
            category = str(result.get('primary_category', 'other')).lower()
            result['primary_category'] = category if category in CATEGORIES else 'other'
            result['confidence'] = normalized_confidence(result.get('confidence'))
            return result
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    return {}


def caption_category(caption: str) -> Optional[str]:
    text = (caption or '').lower()
    scored = {
        category: sum(keyword.lower() in text for keyword in keywords)
        for category, keywords in CATEGORY_FEATURES.items()
    }
    best = max(scored, key=scored.get)
    return best if scored[best] else None


def evidence_score(result: Dict) -> float:
    category = result.get('primary_category', 'other')
    evidence = result.get('visual_evidence', [])
    axes = result.get('visible_axes_text', [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if isinstance(axes, dict):
        axes = list(axes.keys()) + [str(value) for value in axes.values()]
    elif isinstance(axes, str):
        axes = [axes]
    text = ' '.join(map(str, evidence + axes)).lower()
    keywords = CATEGORY_FEATURES.get(category, [])
    hits = sum(keyword.lower() in text for keyword in keywords)
    return min(1.0, 0.45 + 0.18 * hits) if keywords else 0.5


def quality_score(figure: Dict) -> float:
    width = int(figure.get('width') or 0)
    height = int(figure.get('height') or 0)
    shortest = min(width, height)
    if shortest >= 700:
        return 1.0
    if shortest >= 400:
        return 0.85
    if shortest >= 200:
        return 0.65
    return 0.4


def initial_score(figure: Dict, result: Dict) -> Tuple[float, bool]:
    visual_category = result['primary_category']
    text_category = caption_category(str(figure.get('caption', '')))
    consistent = text_category is None or text_category == visual_category
    caption_component = 0.7 if text_category is None else float(consistent)
    score = (
        0.50 * result['confidence']
        + 0.30 * evidence_score(result)
        + 0.10 * caption_component
        + 0.10 * quality_score(figure)
    )
    if not consistent:
        score -= 0.12
    return max(0.0, min(1.0, score)), not consistent


def final_score(figure: Dict, primary: Dict, review: Optional[Dict]) -> Tuple[float, bool]:
    raw, caption_conflict = initial_score(figure, primary)
    if not review:
        return raw, caption_conflict
    primary_secondary = set(primary.get('secondary_categories') or [])
    review_secondary = set(review.get('secondary_categories') or [])
    agreement = (
        primary['primary_category'] == review['primary_category']
        or primary['primary_category'] in review_secondary
        or review['primary_category'] in primary_secondary
    )
    score = (
        0.32 * primary['confidence']
        + 0.23 * review['confidence']
        + 0.20 * float(agreement)
        + 0.15 * evidence_score(primary)
        + 0.10 * quality_score(figure)
    )
    if caption_conflict:
        score -= 0.08
    if not agreement:
        score -= 0.18
    return max(0.0, min(1.0, score)), caption_conflict


def risk_score(figure: Dict) -> int:
    old = str(figure.get('category', 'other'))
    cap = (figure.get('caption') or '').lower()
    score = 0
    if old in {'electrochemistry', 'xrd'}:
        score += 5
    incompatible = {
        'electrochemistry': CATEGORY_FEATURES['theoretical_calculation'] + CATEGORY_FEATURES['electron_microscopy'],
        'xrd': CATEGORY_FEATURES['thermal_analysis'] + CATEGORY_FEATURES['raman_ftir'] + CATEGORY_FEATURES['electron_microscopy'],
    }
    score += 3 * sum(keyword.lower() in cap for keyword in incompatible.get(old, []))
    if not cap.strip():
        score += 2
    if not figure.get('categories'):
        score += 1
    return score


def select_pilot(index: List[Dict], limit: int, seed: int) -> List[Dict]:
    candidates = [figure for figure in index if image_path(figure).exists()]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates.sort(key=risk_score, reverse=True)
    selected = candidates[:limit]
    # Always include the known theoretical-computation false positive when present.
    known_file = '1-s2.0-S0008622320306576-main_p7_img0.png'
    known = next((figure for figure in candidates if figure.get('file') == known_file), None)
    if known and known not in selected:
        selected[-1] = known
    return selected


def write_results(rows: List[Dict]) -> None:
    PILOT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    flat_rows = []
    for row in rows:
        flat_rows.append({
            'file': row['file'], 'pdf': row.get('pdf', ''),
            'old_category': row['old_category'], 'new_category': row['new_category'],
            'confidence': row['confidence'], 'status': row['status'],
            'caption_conflict': row['caption_conflict'],
            'review_used': bool(row.get('review')),
            'caption': row.get('caption', '')[:300],
            'visual_evidence': json.dumps(row['primary'].get('visual_evidence', []), ensure_ascii=False),
        })
    with PILOT_CSV.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260820)
    parser.add_argument('--primary-model', default=PRIMARY_MODEL)
    parser.add_argument('--review-model', default=REVIEW_MODEL)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    index = json.loads(INDEX_PATH.read_text(encoding='utf-8'))
    selected = select_pilot(index, args.limit, args.seed)
    rows = []
    if args.resume and PILOT_JSON.exists():
        rows = json.loads(PILOT_JSON.read_text(encoding='utf-8'))
    completed = {row['file'] for row in rows}
    client = get_client()

    print(f'Selected {len(selected)} high-risk figures; completed={len(completed)}')
    for number, figure in enumerate(selected, 1):
        if figure['file'] in completed:
            continue
        path = image_path(figure)
        prompt = PROMPT.format(caption=(figure.get('caption') or '')[:800])
        try:
            primary = call_model(client, args.primary_model, path, prompt)
            raw_score, caption_conflict = initial_score(figure, primary)
            old_category = str(figure.get('category', 'other'))
            high_risk_change = (
                old_category in {'xrd', 'electrochemistry'}
                and primary['primary_category'] != old_category
            )
            needs_review = raw_score < 0.80 or caption_conflict or high_risk_change
            review = None
            if needs_review:
                review = call_model(
                    client, args.review_model, path,
                    REVIEW_PROMPT.format(categories=', '.join(CATEGORIES)),
                )
            confidence, caption_conflict = final_score(figure, primary, review)
            compatible_review = review and (
                review['primary_category'] == primary['primary_category']
                or review['primary_category'] in set(primary.get('secondary_categories') or [])
                or primary['primary_category'] in set(review.get('secondary_categories') or [])
            )
            if review and not compatible_review:
                new_category = 'other'
                status = 'manual_review'
            else:
                new_category = primary['primary_category']
                status = 'accepted' if confidence >= 0.85 else 'manual_review'
            rows.append({
                'file': figure['file'], 'pdf': figure.get('pdf', ''),
                'path': str(path), 'caption': figure.get('caption', ''),
                'old_category': old_category, 'new_category': new_category,
                'confidence': round(confidence, 4), 'status': status,
                'caption_conflict': caption_conflict,
                'primary': primary, 'review': review,
            })
        except Exception as exc:
            rows.append({
                'file': figure['file'], 'pdf': figure.get('pdf', ''),
                'path': str(path), 'caption': figure.get('caption', ''),
                'old_category': figure.get('category', 'other'),
                'new_category': 'other', 'confidence': 0.0,
                'status': 'api_error', 'caption_conflict': False,
                'primary': {'error': f'{type(exc).__name__}: {str(exc)[:400]}'},
                'review': None,
            })
        write_results(rows)
        print(
            f"[{number}/{len(selected)}] {figure['file'][:45]}: "
            f"{rows[-1]['old_category']} -> {rows[-1]['new_category']} "
            f"({rows[-1]['confidence']:.2f}, {rows[-1]['status']})",
            flush=True,
        )
        time.sleep(0.4)

    changed = sum(row['old_category'] != row['new_category'] for row in rows)
    accepted = sum(row['status'] == 'accepted' for row in rows)
    print(f'Done: {len(rows)} rows, {changed} changed, {accepted} accepted')
    print(f'JSON: {PILOT_JSON}\nCSV: {PILOT_CSV}')


if __name__ == '__main__':
    main()
