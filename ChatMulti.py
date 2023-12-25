import tkinter as tk
from tkinter import messagebox
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型成全局变量
model_name = 'checkpoint/baichuan-7b'
device = 'cuda'
max_new_tokens = 500  # 每轮对话最多生成多少个token
history_max_len = 1000  # 模型记忆的最大token长度
top_p = 0.9
temperature = 0.35
repetition_penalty = 1.0

# 先加载模型，再加载界面，避免界面崩溃
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True
)
# 记录所有历史记录
history_token_ids = tokenizer('<s>', return_tensors="pt").input_ids


def inferAns(user_input):
    global history_token_ids
    if user_input == '':
        return user_input
    user_input = '{}</s>'.format(user_input)
    user_input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids
    history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
    model_input_ids = history_token_ids[:, -history_max_len:].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=model_input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
            temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id
        )
    model_input_ids_len = model_input_ids.size(1)
    response_ids = outputs[:, model_input_ids_len:]
    history_token_ids = torch.concat((history_token_ids, response_ids.cpu()), dim=1)
    response = tokenizer.batch_decode(response_ids)
    return response[0].strip().replace('</s>', "")


def send_message(event=None):
    question = input_box.get("1.0", "end-1c").strip()  # 获取输入框中的文本
    input_box.delete("1.0", tk.END)  # 清空输入框
    input_box.delete("end-2c", tk.END)
    output_box.configure(state=tk.NORMAL)
    if question == '':
        messagebox.showinfo("警告", "请先输入问题！")
        output_box.configure(state=tk.DISABLED)
        return

    output_box.insert(tk.END, f"提问：\n\n{question}\n\n")
    # 将question传递给模型进行推理，获取回答answer
    answer = inferAns(question)
    output_box.insert(tk.END, f"回答：\n\n{answer}\n\n")
    output_box.configure(state=tk.DISABLED)
    output_box.see(tk.END)


# 创建主窗口
window = tk.Tk()
window.title("AI-Chat Rob")
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window_width = 800
window_height = 600
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# 创建输出框
output_box = tk.Text(window, font=("Microsoft YaHei", 14))
output_box.pack(fill=tk.BOTH, expand=True)
output_box.configure(state=tk.DISABLED)

# 创建输入框
input_box = tk.Text(window, height=5, font=("Microsoft YaHei", 14))
input_box.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)
input_box.bind("<Control-Return>", send_message) # 绑定一个crtl+enter事件

# 创建发送按钮
send_button = tk.Button(window, text="发送", command=send_message, height=5)
send_button.pack(side=tk.RIGHT, padx=10, pady=10)

# 运行主循环
window.mainloop()

# 测试的句子
# 你好，清介绍自己
# 我想学好机器学习，该怎么做
# 我可以在哪些网站上进行学习马
# 请你对这句话进行分词：“从1992年开始研制的长征二号F型火箭，是中国航天史上技术最复杂、可靠性和安全性指标最高的运载火箭”，每个词用引号标记
# 请你识别这句话中的命名实体“这个寒假去北京大学玩”，只输出实体名，用逗号分割
# 鲁迅为什么暴打周树人