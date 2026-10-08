# Multimodal Math Solver

A two-stage OCR + LLM pipeline for photo-based mathematical problem recognition and solving.

本项目用于对数学题图片进行识别与自动求解，采用 **OCR + LLM 两阶段架构**。
该项目来源于：

**2025IKCEST第七届“一带一路”国际大数据竞赛暨 第十一届百度&西安交大大数据竞赛 —— 结合大模型的拍照识题与解题赛题**
https://aistudio.baidu.com/competition/detail/1337/0/introduction

## Pipeline

```text
Question Image
      ↓
Ovis2.5-2B
      ↓
Structured OCR Text
      ↓
Qwen2.5-Math-1.5B-Instruct
      ↓
Answer Extraction & Formatting
      ↓
Final Answer
```

支持选择题、填空题、计算应用题、数学公式 LaTeX 结构化识别、批量推理、自动答案提取与格式化及基础异常降级处理。

## 1. Models

### OCR / Vision Model

```text
AIDC-AI/Ovis2.5-2B
```

模型下载后放置于：

```text
./model/
```

### Math Reasoning Model

```text
Qwen/Qwen2.5-Math-1.5B-Instruct
```

模型下载后放置于：

```text
./model_math/
```

> 模型权重不会上传至 GitHub，请使用仓库中的下载脚本自行下载。

## 2. Project Structure

```text
multimodal-math-solver/
│
├── dataset/
│   ├── images/                 # 测试题图片
│   ├── input.jsonl             # 测试输入
│   └── gt.jsonl                # Ground Truth 标准答案
│
├── model/                      # Ovis2.5-2B（下载后生成）
├── model_math/                 # Qwen2.5-Math-1.5B-Instruct（下载后生成）
│
├── download_ovis_model.sh      # 下载 OCR 模型
├── download_qwen_math_model.sh # 下载数学推理模型
│
├── ovis_ocr.py                 # Stage 1：OCR / 图像理解
├── qwen_math.py                # Stage 2：数学推理与答案提取
├── run.py                      # 主推理 Pipeline
├── run.sh                      # Shell 运行入口
├── build_env.sh                # 环境配置脚本
├── requirements.txt            # Python 依赖
├── .gitignore
└── README.md
```

## 3. Installation

建议使用 Linux + NVIDIA GPU 环境运行。

```bash
git clone https://github.com/anything051/multimodal-math-solver.git
cd multimodal-math-solver
pip install -r requirements.txt
```

如需使用仓库提供的环境脚本：

```bash
bash build_env.sh
```

## 4. Download Models

下载 OCR 模型：

```bash
bash download_ovis_model.sh
```

下载数学推理模型：

```bash
bash download_qwen_math_model.sh
```

下载完成后应存在：

```text
model/
model_math/
```

## 5. Quick Test

仓库提供小型测试数据集：

```text
dataset/
├── images/
├── input.jsonl
└── gt.jsonl
```

`dataset/input.jsonl` 每行为一个 JSON 对象，例如：

```json
{"image": "images/example.jpg", "tag": "选择题"}
```

`tag` 支持：

```text
选择题
填空题
计算应用题
```

## 6. Run Test Dataset

在项目根目录运行：

```bash
python run.py "../dataset" "../dataset/input.jsonl" "../dataset/output.jsonl"
```

程序将依次完成：

```text
1. 加载测试输入
2. 图像预处理
3. Ovis OCR / 视觉理解
4. OCR 文本后处理
5. Qwen Math 数学推理
6. 最终答案提取
7. 输出 JSONL 文件
```

## 7. Output Format

输出文件：

```text
result.jsonl
```

示例：

```json
{
  "image": "images/example.jpg",
  "tag": "选择题",
  "step": "",
  "answer": "B"
}
```

填空题和计算应用题会统一格式化为六位小数，例如：

```json
{
  "image": "images/example.jpg",
  "tag": "填空题",
  "step": "",
  "answer": "1.000000"
}
```

## 8. Ground Truth

`dataset/gt.jsonl` 保存测试集对应的标准答案，可与 `result.jsonl` 对比，用于快速验证推理结果。

## 9. OCR Pipeline

OCR 阶段使用 Ovis2.5-2B。

主要流程：

```text
Image
  ↓
Resize to 1536 × 1536
  ↓
RGB Conversion
  ↓
Multimodal Inference
  ↓
Structured OCR Output
```

Prompt 约束包括：

- 完整提取题干
- 提取所有选项
- 保留数学公式
- 转换为标准 LaTeX
- 忽略手写批注
- 忽略页码、水印等无关信息
- 固定结构输出

## 10. Math Reasoning Pipeline

OCR 输出传递给 Qwen2.5-Math-1.5B-Instruct。

模型被要求逐步推理，并将最终答案输出到：

```text
\boxed{}
```

程序随后自动完成：

```text
\boxed{} extraction
        ↓
Multiple-choice Regex Matching
        ↓
LaTeX → Numeric Conversion
        ↓
Answer Validation
        ↓
Final Formatting
```

## 11. Batch Inference

当前主流程：

```text
OCR batch size: 8
Math batch size: 16
```

可根据显存大小调整。

## 12. Error Handling

系统包含基础异常处理机制：

- 图片读取失败时生成默认 RGB 图片
- OCR 推理失败时返回降级标记
- 数学答案提取失败时保留可解析结果
- 非法选择题答案进行格式检查
- 数值结果统一格式化为六位小数

## 13. Requirements

主要依赖：

```text
Python
PyTorch
Transformers
Accelerate
Pillow
qwen-vl-utils
latex2sympy2
regex
tqdm
```

完整依赖请参考 `requirements.txt`。

## 14. Notes

### Model Weights

模型权重体积较大，不提交到 GitHub。

请运行：

```bash
bash download_ovis_model.sh
bash download_qwen_math_model.sh
```

### Dataset

仓库仅提供少量测试样例，用于：

- Pipeline 测试
- 项目展示
- 代码复现验证

不包含完整竞赛评测数据。

### Hardware

建议使用 NVIDIA GPU。

若出现 CUDA OOM，可优先降低 OCR / Math 的 batch size。

## 15. Competition

该项目来源于：

**2025IKCEST第七届“一带一路”国际大数据竞赛暨 第十一届百度&西安交大大数据竞赛 —— 结合大模型的拍照识题与解题赛题**

主要优化包括：

- OCR 模型替换
- Prompt Engineering
- 数学公式 LaTeX 规范化
- Batch Inference
- Answer Extraction
- Error Handling
- Output Validation

实验过程中，算法得分由 `40.0` 提升至 `67.2`，整体提升约 `68%`。

## 16. License

本仓库仅提供项目代码及少量测试数据。

Ovis2.5 和 Qwen2.5-Math 模型权重请遵循各自官方 License。
