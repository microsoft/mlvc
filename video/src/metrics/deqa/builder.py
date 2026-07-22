import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor


def load_pretrained_model(model_path: str):
    kwargs = {"torch_dtype": torch.float16}

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, **kwargs)
    image_processor = CLIPImageProcessor.from_pretrained(model_path)

    return tokenizer, model, image_processor
