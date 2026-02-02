from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

contaminated = {
    # LLaVA lora requires 'lora' in its path
    "llava-lora": partial(LLaVA, model_path="path/to/model_lora"),
    "llava-fft": partial(LLaVA, model_path="path/to/model"),

    # Qwen: LoRA (adapter) vs full fine-tune (fft)
    "qwen-lora": partial(Qwen2VLChat, model_path="/path/to/qwen_lora", min_pixels=3 * 512 * 512, max_pixels=3 * 1024 * 1024),
    "qwen-full": partial(Qwen2VLChat, model_path="/path/to/qwen_full_fft", min_pixels=3 * 512 * 512, max_pixels=3 * 1024 * 1024),
}

supported_VLM = contaminated