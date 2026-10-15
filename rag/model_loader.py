import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_local_model():
    base_path = os.getenv("BASE_MODEL_PATH")
    adapter_path = os.getenv("ADAPTER_PATH")
    if not base_path or not adapter_path:
        raise RuntimeError("Set BASE_MODEL_PATH and ADAPTER_PATH in .env to directories that exist on this machine.")
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base model directory not found: {base_path}")
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=quant_config, device_map="auto")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("Fine-tuned Llama model loaded.")
    return tokenizer, model
