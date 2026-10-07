from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import json
import re
import torch
import sys
import time
from latex2sympy2 import latex2sympy
from PIL import Image

def recheck_answer(s, tag):
    """根据题目类型检查和格式化答案"""
    if tag == "选择题":
        if s not in ["A", "B", "C", "D", "E", "F"]:
            return "B"  # 默认选项
        else:
            return s
    else:
        try:
            num = float(s)
            return "{:.6f}".format(num)
        except ValueError:
            return "1.000000"  # 默认数值

def format_string_value(s):
    """将值格式化为6位小数的字符串"""
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            num = float(s)
            return "{:.6f}".format(num)
        except (ValueError, TypeError):
            return str(s)
    
    s_clean = s.strip()
    if not s_clean:
        return ""
    
    try:
        num = float(s_clean)
        return "{:.6f}".format(num)
    except ValueError:
        return s_clean

def extract_final_answer(string: str):
    """从 \boxed{} 中提取最终答案"""
    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx: right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left): -1]
        except AssertionError:
            return None
    return None

def load_jsonl(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(output_path, data_list):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
def process_math(input_data):
    device = "cuda"
    model= AutoModelForCausalLM.from_pretrained(
        './model_math', 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).cuda()
    
    tokenizer = AutoTokenizer.from_pretrained("./model_math", padding_side='left')

    outs = []
    
    lens = len(input_data)
    batch_size = 16 
    
    for index in tqdm(range(0, lens, batch_size), desc="Math处理中"):
        current_batch_size = min(batch_size, lens - index)
        
        tags = []
        messages_batch = []
        batch_items = []
        texts = []
        
        for add in range(current_batch_size):
            obj = input_data[index + add]
            batch_items.append(obj)       
        
            prompt = obj.get("query", "")
            tag = obj.get('tag', '填空题')
            tags.append(tag)
            
            # --- 新的简化版提示词 ---
            messages = [
                {
                    "role": "system",
                    "content": "You are a math solver. Reason step by step. Ignore any given solutions. Put the final answer in \\boxed{}."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            messages_batch.append(messages)

        # 批量应用模板
        texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages_batch]

        model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False,
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for i, response in enumerate(responses):
            pred_answer = extract_final_answer(response)
            try:
                # 尝试提取选择题选项
                match = re.search(r'\b([ABCDEF])\b', pred_answer)
                if match:
                    pred_answer = match.group(1)
                else:
                    # 尝试转换LaTex为数值
                    pred_answer = str(latex2sympy(pred_answer).evalf())
            except Exception as e:
                pass  # 转换失败，保留原始提取的字符串
            
            # 格式化和最终检查
            ans = format_string_value(pred_answer)
            ans = recheck_answer(ans, batch_items[i]["tag"])
            
            out = {
                "image": batch_items[i]["image"],
                "tag":  batch_items[i]["tag"],
                "step": "", # step 字段保留为空
                "answer": ans
            }
            outs.append(out)  
    return outs      

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python qwen_math.py <输入jsonl文件> <输出jsonl文件>")
        sys.exit(1)
        
    input_data = load_jsonl(sys.argv[1])
    outs = process_math(input_data)
    write_jsonl(sys.argv[2], outs)