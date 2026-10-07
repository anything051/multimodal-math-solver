from transformers import AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import json
import re
import torch
import sys
from PIL import Image

def build_output_prompt(tag):
    """构建OCR查询，附加到清理后的模型响应中，供下一阶段使用"""
    if tag == "选择题":
        return "The current question is a multiple-choice question. Extract ALL options exactly as shown, including option labels (A, B, C, D, etc.)."
    elif tag == "填空题":
        return "The current question is a fill-in-the-blank question. Focus on identifying the blank positions and required format."
    else:
        return "The current question is a calculation word problem. Extract the complete problem statement with all numerical values and relationships."

def load_jsonl(input_file):
    """加载jsonl文件"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        return data
    except Exception as e:
        print(f"CRITICAL: 加载文件失败: {e}")
        raise

def write_jsonl(output_path, data_list):
    """写入jsonl文件"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"CRITICAL: 写入文件失败: {e}")
        raise

def resize_image(image_path, target_size=1536):
    """调整图像大小并转换为RGB"""
    try:
        image = Image.open(image_path)
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        print(f"警告: 图像处理失败 {image_path}: {e}")
        return Image.new('RGB', (target_size, target_size), color='white')

def create_optimized_prompt():
    """创建优化的提示词 (新版 - 严格准确版)"""
    return """
**ROLE:** You are a high-precision OCR engine specialized in mathematical content.

**TASK:** Your sole and exclusive task is to perform a VERBATIM transcription of the mathematical problem from the image.
- You must NOT solve the problem.
- You must NOT interpret, rephrase, or add any explanatory text.

**CRITICAL INSTRUCTIONS:**

1.  **VERBATIM EXTRACTION:**
    - Transcribe ALL printed text, numbers, and symbols that constitute the problem.
    - This includes:
        1.  The full question statement.
        2.  All equations, formulas, and expressions.
        3.  All multiple-choice options, including their exact labels (e.g., "A.", "B)", "(C)").
        4.  All text from diagrams, charts, or graphs (e.g., axis labels, units, titles).

2.  **MATHEMATICAL NOTATION (Most Important):**
    - You MUST preserve mathematical notation using standard, machine-readable LaTeX format.
    - **Fractions:** Use `\frac{numerator}{denominator}` (e.g., `\frac{1}{2}`).
    - **Exponents/Subscripts:** Use `^` and `_` (e.g., `x^2`, `H_2O`).
    - **Square Roots:** Use `\sqrt{x}` (e.g., `\sqrt{3}`).
    - **Special Symbols:** Use standard LaTeX commands (e.g., `\pi`, `\theta`, `\alpha`, `\ge`, `\le`, `\neq`).
    - **Matrices/Systems of Equations:** Preserve the layout and structure.

3.  **STRICTLY IGNORE:**
    - **ABSOLUTELY IGNORE** all handwritten content, including solutions, answers, checkmarks, notes, or calculations.
    - Ignore page numbers, headers, footers, or watermarks unless they are part of the problem logic.
    - Ignore stray marks, smudges, or artifacts.

**STRICT OUTPUT FORMAT:**
Provide your response ONLY in the following format. Do not add any text before "Problem:".

Problem: [Insert complete, verbatim transcription of the problem text and equations here, using LaTeX for all mathematical notation]
Options: [Insert all multiple-choice options here, one per line. If no options are present, write "None"]
"""

def clean_model_response(response, tag):
    """清理模型的原始响应 (适配新的严格提示词)"""
    
    # 移除新提示词中的所有指导性文本
    patterns_to_remove = [
        r'\*\*ROLE:\*\*.*?STRICT OUTPUT FORMAT:',
        r'\[Insert complete, verbatim transcription.*?\]',
        r'\[Insert all multiple-choice options.*?\]',
    ]
    
    for pattern in patterns_to_remove:
        response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # 对于非选择题，移除 "Options:" 标签
    if tag != "选择题":
        response = re.sub(r'Options:.*', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # 清理掉 "Options: None" (如果存在)
    response = re.sub(r'Options:\s*None\s*', '', response, flags=re.IGNORECASE)
    
    response = re.sub(r'\n\s*\n', '\n', response)  # 合并多余空行
    return response.strip()

def process_batch(model, batch_data, image_dir, question):
    """批量处理图像"""
    batch_size = len(batch_data)
    
    messages_batch = []
    tags = []
    image_paths = []
    
    for obj in batch_data:
        image_path = os.path.join(image_dir, obj['image'])
        tag = obj.get('tag', '填空题')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resize_image(image_path)},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        messages_batch.append(messages)
        tags.append(tag)
        image_paths.append(obj['image'])
    
    try:
        input_ids_batch, pixel_values_batch, grid_thws_batch = [], [], []
        
        max_tokens = 1024
        thinking_budget = 512
        enable_thinking = False
        enable_thinking_budget = False
        
        for messages in messages_batch:
            input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            input_ids_batch.append(input_ids)
            pixel_values_batch.append(pixel_values)
            grid_thws_batch.append(grid_thws)
        
        input_ids = torch.cat(input_ids_batch, dim=0).cuda()
        pixel_values = torch.cat(pixel_values_batch, dim=0).cuda().to(model.dtype) if pixel_values_batch[0] is not None else None
        grid_thws = torch.cat(grid_thws_batch, dim=0).cuda() if grid_thws_batch[0] is not None else None
        
        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                enable_thinking=enable_thinking,
                enable_thinking_budget=enable_thinking_budget,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=model.text_tokenizer.eos_token_id,
                pad_token_id=model.text_tokenizer.pad_token_id,
                thinking_budget=thinking_budget,
            )
        
        results = []
        for i in range(batch_size):
            response = model.text_tokenizer.decode(outputs[i], skip_special_tokens=True)
            tag = tags[i]
            cleaned_response = clean_model_response(response, tag)
            
            output_entry = {
                "image": image_paths[i],
                "query": build_output_prompt(tag) + "\n" + cleaned_response, # 组合下一阶段的提示词和OCR结果
                "tag": tag,
            }
            results.append(output_entry)
            
        return results
        
    except Exception as e:
        print(f"CRITICAL: 批量处理失败: {e}")
        return [{
            "image": image_paths[i],
            "query": "OCR failed", # 提供一个失败的query
            "tag": tags[i],
        } for i in range(batch_size)]

def process_ocr(image_dir, input_data, batch_size=4):
    """主处理函数"""
    question = create_optimized_prompt()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            './model', 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            device_map="auto"
        )
        model.eval()
        print("OCR模型加载成功")
    except Exception as e:
        print(f"CRITICAL: OCR模型加载失败: {e}")
        raise
    
    output_lines = []
    
    for i in tqdm(range(0, len(input_data), batch_size), desc="OCR处理中"):
        batch_data = input_data[i:i+batch_size]
        batch_results = process_batch(model, batch_data, image_dir, question)
        output_lines.extend(batch_results)
    
    print(f"OCR处理完成，共 {len(output_lines)} 条数据")
    return output_lines


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python ovis_ocr.py <图片目录> <输入jsonl文件> <输出jsonl文件>")
        sys.exit(1)
    
    try:
        input_data = load_jsonl(sys.argv[2])
        output_lines = process_ocr(sys.argv[1], input_data, batch_size=4)
        write_jsonl(sys.argv[3], output_lines)
        
    except Exception as e:
        print(f"CRITICAL: 程序执行失败: {e}")
        sys.exit(1)

    print("OCR任务全部完成。")