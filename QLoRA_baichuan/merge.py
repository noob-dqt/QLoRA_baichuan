from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
"""
使用该脚本，将lora的权重合并到base model中
"""

def merge_lora_to_base_model():
    model_name_or_path = 'baichuan-inc/baichuan-7B'
    adapter_name_or_path = 'output/baichuan-7b/checkpoint-13000'    # adapter_config.json
    save_path = 'checkpoint/baichuan-7b'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

if __name__ == '__main__':
    merge_lora_to_base_model()