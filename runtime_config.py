"""Project paths and first-run initialization for the student edition."""

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
SAMPLE_DIR = PROJECT_ROOT / 'data' / 'sample'


def initialize_sample_data() -> None:
    """Install missing sample inputs while preserving existing local data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ('structured_records.csv', 'knowledge_base.json',
                 'figure_index.json', 'db_summary_for_recommendation.md'):
        target = OUTPUT_DIR / name
        if not target.exists():
            shutil.copyfile(SAMPLE_DIR / name, target)


def api_config() -> tuple[str, str]:
    """Read credentials only from environment variables."""
    key = os.getenv('DEEPSEEK_API_KEY', '').strip()
    url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com').strip()
    if not key:
        raise ValueError('请设置环境变量 DEEPSEEK_API_KEY，然后重启应用。')
    return key, url or 'https://api.deepseek.com'
