from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 单轮对话
model_name = 'checkpoint/baichuan-7b'
max_new_tokens = 500    
top_p = 0.9
temperature = 0.35
repetition_penalty = 1.0
device = 'cuda'
input_pattern = '<s>{}</s>'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
    #device_map='auto'
).half().to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True
)
text = input('提问：')
while True:
    if text=='quit':
        break
    text = text.strip()
    text = input_pattern.format(text)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    response = response.strip().replace(text, "").replace('</s>', "").replace('<s>', "").strip()
    print("回答：{}".format(response))
    text = input('提问：')