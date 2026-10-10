import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model_path_utils import QWEN_CACHE_DIR_NAME, QWEN_MODEL_ID, default_hf_cache_root, resolve_hf_model_path


SYSTEM_PROMPT = (
    'You are a CO2 electrocatalysis IE assistant. '
    'Return only strict JSON with keys: materials, products, efficiency. '
    'products must be chemical names only (e.g. CO, HCOOH, CH4, formate). '
    "efficiency format: 'X% [at -Y V vs RHE]'. "
    'Each value must be an array of strings. Do not output any extra text.'
)

RETRY_SUFFIXES = [
    '',
    '\nIMPORTANT: Return JSON only. No markdown, no explanation.',
    '\nIMPORTANT: Output exactly this shape: {"materials": [], "products": [], "efficiency": []}',
]


def load_adapter_tokenizer(adapter_path: str):
    """Preserve saved special tokens across Transformers tokenizer formats."""
    kwargs = {'local_files_only': True}
    config_path = Path(adapter_path) / 'tokenizer_config.json'
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding='utf-8'))
        extra = config.get('extra_special_tokens')
        if isinstance(extra, list):
            # Transformers 4 expects a dict here; v5 saves a list instead.
            kwargs['extra_special_tokens'] = {}
            kwargs['additional_special_tokens'] = extra
    return AutoTokenizer.from_pretrained(adapter_path, **kwargs)


def parse_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    return mapping[name]


def build_prompt(instruction: str, context: str, retry_note: str = '') -> str:
    user_text = f'{instruction}\n\nContext:\n{context}{retry_note}'
    return (
        '<|system|>\n'
        f'{SYSTEM_PROMPT}<|end|>\n'
        '<|user|>\n'
        f'{user_text}<|end|>\n'
        '<|assistant|>\n'
    )


def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidate = text.strip()
    if '<|end|>' in candidate:
        candidate = candidate.split('<|end|>', 1)[0].strip()
    left = candidate.find('{')
    right = candidate.rfind('}')
    if left == -1 or right == -1 or right <= left:
        return None
    candidate = candidate[left:right + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_schema(obj: Dict[str, Any]) -> Tuple[bool, str]:
    required = ['materials', 'products', 'efficiency']
    for key in required:
        if key not in obj:
            return False, f'missing key: {key}'
        if not isinstance(obj[key], list):
            return False, f'key "{key}" is not a list'
        if not all(isinstance(x, str) for x in obj[key]):
            return False, f'key "{key}" must contain only strings'
    return True, ''


def generate_once(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_input_length).to(model.device)
    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def predict_with_retry(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    instruction: str,
    context: str,
    max_retries: int,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    attempts: List[Dict[str, str]] = []
    for i in range(max_retries + 1):
        retry_note = RETRY_SUFFIXES[min(i, len(RETRY_SUFFIXES) - 1)]
        prompt = build_prompt(instruction, context, retry_note)
        raw = generate_once(model, tokenizer, prompt, max_input_length, max_new_tokens, temperature)
        parsed = extract_json_obj(raw)
        if parsed is not None:
            ok, err = validate_schema(parsed)
            if ok:
                return {
                    'success': True,
                    'attempt': i + 1,
                    'result': parsed,
                    'raw_output': raw,
                    'attempt_logs': attempts,
                }
            attempts.append({'attempt': str(i + 1), 'error': err, 'raw': raw[:300]})
        else:
            attempts.append({'attempt': str(i + 1), 'error': 'invalid JSON', 'raw': raw[:300]})

    return {
        'success': False,
        'attempt': max_retries + 1,
        'result': {'materials': [], 'products': [], 'efficiency': []},
        'raw_output': attempts[-1]['raw'] if attempts else '',
        'attempt_logs': attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='LoRA inference with JSON validation and auto-retry.')
    parser.add_argument('--base_model', default=QWEN_CACHE_DIR_NAME)
    parser.add_argument('--cache_dir', default=str(default_hf_cache_root()))
    parser.add_argument('--adapter_path', default='lora_checkpoints_qwen25_7b_qlora_json')
    parser.add_argument('--instruction', default='Extract electrocatalyst information from the text.')
    parser.add_argument('--context', required=True, help='Input text context.')
    parser.add_argument('--max_retries', type=int, default=2)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--max_new_tokens', type=int, default=180)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--load_in_4bit', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--bnb_4bit_quant_type', default='nf4', choices=['nf4', 'fp4'])
    parser.add_argument('--bnb_4bit_compute_dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--trust_remote_code', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    resolved_base_model = resolve_hf_model_path(args.base_model, fallback_model_id=QWEN_MODEL_ID, cache_dir=args.cache_dir)
    print(f'Using base model: {resolved_base_model}')
    print(f'Using cache dir: {args.cache_dir}')

    tokenizer = load_adapter_tokenizer(args.adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        'device_map': 'auto',
        'trust_remote_code': args.trust_remote_code,
    }
    if args.load_in_4bit:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=parse_torch_dtype(args.bnb_4bit_compute_dtype),
        )
    else:
        dtype = parse_torch_dtype(args.bnb_4bit_compute_dtype)
        model_kwargs['torch_dtype'] = dtype if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(resolved_base_model, cache_dir=args.cache_dir, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    result = predict_with_retry(
        model=model,
        tokenizer=tokenizer,
        instruction=args.instruction,
        context=args.context,
        max_retries=args.max_retries,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
