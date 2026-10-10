"""Lazy, reusable local LoRA extraction with the original training prompt."""

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
_EXTRACTORS = {}


def configured_path(profile: dict, name: str, default: str = '') -> str:
    env_name = profile.get(f'{name}_env', '')
    value = os.getenv(env_name, '') if env_name else ''
    value = value.strip() or str(profile.get(name) or default)
    path = Path(value).expanduser()
    if value and not path.is_absolute() and (PROJECT_ROOT / path).exists():
        return str((PROJECT_ROOT / path).resolve())
    return value


class LocalLoraExtractor:
    """Load weights once per profile, without downloading models implicitly."""

    def __init__(self, profile: dict):
        adapter = configured_path(profile, 'adapter_path')
        adapter_dir = Path(adapter).expanduser()
        if not adapter or not (adapter_dir / 'adapter_config.json').is_file():
            raise ValueError('LoRA adapter 不存在；请设置 CO2RR_LORA_ADAPTER_PATH 或 primary.adapter_path。')
        if not any((adapter_dir / name).is_file() for name in
                   ('adapter_model.safetensors', 'adapter_model.bin')):
            raise ValueError('LoRA adapter 缺少权重文件。权重应单独存放，不包含在学生版中。')
        config = json.loads((adapter_dir / 'adapter_config.json').read_text())
        base_ref = configured_path(profile, 'base_model')
        if not base_ref:
            trained_base = str(config.get('base_model_name_or_path', ''))
            if Path(trained_base).is_dir():
                base_ref = trained_base
            elif trained_base and not Path(trained_base).is_absolute():
                base_ref = trained_base
            else:
                raise ValueError('原训练基础模型路径在本机不存在；请设置 CO2RR_LORA_BASE_MODEL。')
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            from inference_lora_json import build_prompt, load_adapter_tokenizer, predict_with_retry
            from model_path_utils import resolve_hf_model_path
        except ImportError as exc:
            raise RuntimeError('本地 LoRA 依赖缺失，请安装 requirements-model.txt。') from exc

        cache_dir = os.getenv('HF_HUB_CACHE') or profile.get('cache_dir')
        base_ref = resolve_hf_model_path(base_ref, cache_dir=cache_dir)
        kwargs = {'device_map': profile.get('device_map', 'auto'),
                  'local_files_only': True, 'trust_remote_code': False}
        dtype_name = profile.get('dtype', 'float16')
        if dtype_name not in ('float16', 'bfloat16', 'float32'):
            raise ValueError('LoRA dtype 必须为 float16、bfloat16 或 float32。')
        dtype = getattr(torch, dtype_name) if torch.cuda.is_available() else torch.float32
        if profile.get('load_in_4bit', False):
            if not torch.cuda.is_available():
                raise ValueError('4-bit LoRA 推理需要 CUDA；CPU 请设置 load_in_4bit=false。')
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            )
        else:
            kwargs['torch_dtype'] = dtype
        self.tokenizer = load_adapter_tokenizer(str(adapter_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(base_ref, cache_dir=cache_dir, **kwargs)
        self.model = PeftModel.from_pretrained(base_model, str(adapter_dir), local_files_only=True)
        self.model.eval()
        self.adapter_path = adapter_dir.resolve()
        self.base_model = base_ref
        self.profile = profile
        self.build_prompt = build_prompt
        self.predict = predict_with_retry

    def extract(self, text: str) -> tuple[list[dict], dict]:
        instruction = ('Extract electrocatalyst information from the text. '
                       'Return only valid JSON with keys materials, products, efficiency. '
                       'Each key must map to an array of strings.')
        input_limit = int(self.profile.get('max_input_length', 1024))
        overlap = int(self.profile.get('chunk_overlap_tokens', 96))
        # Reserve space for the longest retry instruction; never truncate silently.
        reserve = '\nIMPORTANT: Output exactly this shape: {"materials": [], "products": [], "efficiency": []}'
        overhead = len(self.tokenizer.encode(self.build_prompt(instruction, '', reserve)))
        budget = input_limit - overhead - 32
        if budget < 128 or not 0 <= overlap < budget:
            raise ValueError('LoRA 输入长度太短或 chunk_overlap_tokens 超出有效上下文长度。')
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + budget, len(tokens))
            chunk = self.tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            # A tokenizer round trip can increase token count; shrink if necessary.
            while len(self.tokenizer.encode(self.build_prompt(instruction, chunk, reserve))) > input_limit:
                end -= 16
                if end <= start + overlap:
                    raise ValueError('LoRA 分块无法容纳在模型上下文中。')
                chunk = self.tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            chunks.append(chunk)
            if end == len(tokens):
                break
            start = end - overlap
        if len(chunks) > int(self.profile.get('max_chunks', 64)):
            raise ValueError(f'文献需要 {len(chunks)} 个 LoRA 分块，超过 max_chunks；未截断文献。')
        outputs = []
        for chunk in chunks:
            result = self.predict(
                self.model, self.tokenizer, instruction, chunk,
                int(self.profile.get('max_retries', 2)), input_limit,
                int(self.profile.get('max_new_tokens', 256)), 0.0,
            )
            if not result['success']:
                raise ValueError('LoRA 未返回有效提取 JSON。')
            outputs.append({'source_text': chunk, 'result': result['result']})
        return outputs, {'backend': 'local_lora', 'chunks': len(chunks),
                         'input_tokens': len(tokens), 'api_calls': 0,
                         'model': f'LoRA:{self.adapter_path.name}',
                         'adapter_path': str(self.adapter_path),
                         'base_model': self.base_model}


def extract_with_lora(profile: dict, text: str) -> tuple[list[dict], dict]:
    cache_key = json.dumps(profile, sort_keys=True) + '\n' + '\n'.join(
        os.getenv(profile.get(f'{name}_env', ''), '')
        for name in ('adapter_path', 'base_model')
    )
    if cache_key not in _EXTRACTORS:
        _EXTRACTORS[cache_key] = LocalLoraExtractor(profile)
    return _EXTRACTORS[cache_key].extract(text)
