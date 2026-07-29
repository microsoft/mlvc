import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessorPil


def load_pretrained_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
    )
    image_processor = CLIPImageProcessorPil.from_pretrained(model_path)

    return tokenizer, model, image_processor
