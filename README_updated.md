# Multimodal Math Solver

一个两阶段的数学题图片识别与自动求解项目：

**题目图片 → OCR 文本提取 → 数学推理 → 答案解析与格式化**

项目使用：

- **OCR 模型**：`AIDC-AI/Ovis2.5-2B`
- **数学推理模型**：`Qwen/Qwen2.5-Math-1.5B-Instruct`

> 模型权重不直接提交到 GitHub。请使用仓库中的下载脚本获取模型。

## 1. 项目结构

```text
submission/
├── model/                         # OCR 模型目录（脚本下载）
├── model_math/                    # 数学模型目录（脚本下载）
│
├── download_ovis_model.sh         # 下载 Ovis2.5-2B 到 ./model/
├── download_qwen_math_model.sh    # 下载 Qwen2.5-Math-1.5B-Instruct 到 ./model_math/
│
├── ovis_ocr.py                    # 阶段一：OCR
├── qwen_math.py                   # 阶段二：数学推理
├── run.py                         # 主流程
├── run.sh                         # 一键运行脚本
├── build_env.sh                   # 环境构建脚本
├── requirements.txt               # Python 依赖
└── README.md
```

代码默认模型目录：

```text
OCR 模型：  ./model/
数学模型： ./model_math/
```

## 2. 环境准备

建议使用 Linux + NVIDIA GPU 环境。

安装项目依赖：

```bash
pip install -r requirements.txt
```

下载脚本会使用 `huggingface_hub`。如果本地没有安装，脚本会自动安装。

## 3. 下载模型

### 3.1 下载 OCR 模型

```bash
bash download_ovis_model.sh
```

模型将保存到：

```text
./model/
```

对应 Hugging Face 模型：

```text
AIDC-AI/Ovis2.5-2B
```

### 3.2 下载数学推理模型

```bash
bash download_qwen_math_model.sh
```

模型将保存到：

```text
./model_math/
```

对应 Hugging Face 模型：

```text
Qwen/Qwen2.5-Math-1.5B-Instruct
```

如果 Hugging Face 网络访问受限，可先配置可用的 Hugging Face 镜像或代理，再执行下载脚本。

## 4. 运行方式

主程序调用格式：

```bash
python run.py <image_dir> <input_jsonl> <output_jsonl>
```

示例：

```bash
python run.py ./images ./sample.jsonl ./result.jsonl
```

也可以根据比赛/服务器环境配置好变量后运行：

```bash
bash run.sh
```

## 5. 输入格式

输入文件为 JSONL，每行一条样本，例如：

```json
{"image":"example_1.png","tag":"选择题"}
{"image":"example_2.png","tag":"填空题"}
{"image":"example_3.png","tag":"计算应用题"}
```

其中：

- `image`：题目图片相对于 `image_dir` 的路径
- `tag`：题型，目前支持 `选择题`、`填空题`、`计算应用题`

## 6. 两阶段 Pipeline

### Stage 1：OCR

`ovis_ocr.py` 使用 **Ovis2.5-2B**：

1. 加载 `./model/`
2. 将题目图片统一处理为 RGB 图像
3. 使用针对数学内容优化的 Prompt 提取题干、公式和选项
4. 将数学符号尽量规范为 LaTeX
5. 根据题型增加下游数学求解提示
6. 输出结构化 OCR 文本

### Stage 2：Math

`qwen_math.py` 使用 **Qwen2.5-Math-1.5B-Instruct**：

1. 加载 `./model_math/`
2. 批量输入 OCR 结果
3. 进行数学推理
4. 要求最终答案输出在 `\\boxed{}` 中
5. 自动提取最终答案
6. 对选择题进行 A-F 选项匹配
7. 对数值题进行 LaTeX 转换与六位小数格式化

最终结果由 `run.py` 写入 JSONL 文件。

## 7. 输出格式

示例：

```json
{"image":"example_1.png","tag":"选择题","step":"","answer":"B"}
{"image":"example_2.png","tag":"填空题","step":"","answer":"3.141593"}
```

字段说明：

- `image`：原始图片路径
- `tag`：题型
- `step`：当前版本保留为空
- `answer`：最终答案

## 8. 技术栈

- Python
- PyTorch
- Transformers
- Ovis2.5-2B
- Qwen2.5-Math-1.5B-Instruct
- PIL
- Regular Expression
- latex2sympy2
- JSON / JSONL

## 9. 项目特点

- OCR 与数学推理解耦的两阶段架构
- 针对数学题 OCR 的结构化 Prompt Engineering
- 数学公式 LaTeX 规范化
- Batch 推理
- `\\boxed{}` 自动答案提取
- 正则表达式与 LaTeX 数值后处理
- OCR / 答案解析异常时的降级机制

## 10. 模型与许可证

本仓库只包含项目代码，不包含模型权重。

模型权重请分别从其官方仓库下载，并遵守对应模型许可证：

- `AIDC-AI/Ovis2.5-2B`
- `Qwen/Qwen2.5-Math-1.5B-Instruct`

如果将本项目公开到 GitHub，建议在 `.gitignore` 中加入：

```gitignore
model/
model_math/
*.safetensors
*.bin
*.pt
*.pth
```

避免误将大模型权重提交到 GitHub。
