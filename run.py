import os
from tqdm import tqdm
import json
import re
import torch
import sys
from ovis_ocr import process_ocr
from qwen_math import process_math

def load_jsonl(input_file):
    """加载jsonl文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(output_path, data_list):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python run.py <图片目录> <输入jsonl文件> <输出jsonl文件>")
        sys.exit(1)
        
    image_dir = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    

    all_data = load_jsonl(sys.argv[2])
    data_cnt = len(all_data)
        
    input_data = all_data[0:data_cnt]
    rest_data = all_data[data_cnt:]
    
    ocr_outs = process_ocr(image_dir, input_data, batch_size=8)
    
    math_outs = process_math(ocr_outs)
    write_jsonl(output_file, math_outs)
    
    print(f"处理完成，结果已保存到 {output_file}")