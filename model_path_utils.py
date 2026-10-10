import json
import os
from pathlib import Path
from typing import List, Optional


HF_CACHE_ROOT = Path.home() / '.cache' / 'huggingface' / 'hub'
SHARED_HF_CACHE_ROOT = HF_CACHE_ROOT
QWEN_CACHE_DIR_NAME = 'models--Qwen--Qwen2.5-7B-Instruct'
QWEN_MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'


def _latest_snapshot_dir(path: Path) -> Optional[Path]:
    if path.name == 'snapshots' and path.is_dir():
        snapshots = sorted((p for p in path.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        return snapshots[0] if snapshots else None

    snapshots_dir = path / 'snapshots'
    if not snapshots_dir.is_dir():
        return None

    main_ref = path / 'refs' / 'main'
    if main_ref.is_file():
        commit_hash = main_ref.read_text(encoding='utf-8').strip()
        commit_snapshot = snapshots_dir / commit_hash
        if commit_snapshot.is_dir():
            return commit_snapshot

    snapshots = sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0] if snapshots else None


def _all_shards_exist(snapshot_dir: Path, index_file: str) -> bool:
    index_path = snapshot_dir / index_file
    if not index_path.is_file():
        return False

    try:
        index_data = json.loads(index_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False

    weight_map = index_data.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map:
        return False

    shard_files = {Path(filename).name for filename in weight_map.values()}
    return all((snapshot_dir / shard_file).is_file() for shard_file in shard_files)


def _has_model_weights(path: Path) -> bool:
    if not path.is_dir():
        return False

    if (path / 'model.safetensors').is_file() or (path / 'pytorch_model.bin').is_file():
        return True

    if _all_shards_exist(path, 'model.safetensors.index.json'):
        return True

    if _all_shards_exist(path, 'pytorch_model.bin.index.json'):
        return True

    return False


def has_complete_model_weights(path: Path) -> bool:
    return _has_model_weights(path)


def model_id_to_cache_dir_name(model_id: str) -> str:
    return f'models--{model_id.replace("/", "--")}'


def default_hf_cache_root() -> Path:
    env_cache = os.getenv('HF_HUB_CACHE')
    if env_cache:
        return Path(env_cache).expanduser()

    if SHARED_HF_CACHE_ROOT.parents[1].is_dir():
        return SHARED_HF_CACHE_ROOT

    return HF_CACHE_ROOT


def known_hf_cache_roots(cache_dir: Optional[str] = None) -> List[Path]:
    roots: List[Path] = []

    if cache_dir:
        roots.append(Path(cache_dir).expanduser())

    env_cache = os.getenv('HF_HUB_CACHE')
    if env_cache:
        roots.append(Path(env_cache).expanduser())

    roots.append(default_hf_cache_root())
    roots.append(HF_CACHE_ROOT)

    deduped_roots: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped_roots.append(root)
    return deduped_roots


def resolve_hf_model_path(model_ref: str, fallback_model_id: Optional[str] = None, cache_dir: Optional[str] = None) -> str:
    model_path = Path(model_ref).expanduser()
    candidates: List[Path] = [model_path] if model_path.is_absolute() else [Path.cwd() / model_path, model_path]

    for cache_root in known_hf_cache_roots(cache_dir=cache_dir):
        candidates.append(cache_root / model_path)

    if '/' in model_ref and not model_ref.startswith('models--'):
        cache_dir_name = model_id_to_cache_dir_name(model_ref)
        for cache_root in known_hf_cache_roots(cache_dir=cache_dir):
            candidates.append(cache_root / cache_dir_name)

    for candidate in candidates:
        if not candidate.exists():
            continue
        snapshot_dir = _latest_snapshot_dir(candidate)
        resolved_path = snapshot_dir if snapshot_dir else candidate
        if _has_model_weights(resolved_path):
            return str(resolved_path)
        if fallback_model_id:
            return fallback_model_id
        return str(resolved_path)

    if fallback_model_id and model_ref in {QWEN_CACHE_DIR_NAME, model_id_to_cache_dir_name(fallback_model_id)}:
        return fallback_model_id
    return model_ref
