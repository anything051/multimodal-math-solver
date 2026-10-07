项目代码说明 

本项目是一个两阶段的 AI 流水线，用于从数学题目图片中提取文本并进行求解。

重要：本项目压缩包已包含所有模型文件，复现时无需联网下载模型。

阶段一 (OCR): 使用 ./model 目录中的 Ovis2.5-2B 视觉模型进行文字识别。

阶段二 (Math): 使用 ./model_math 目录中的 Qwen2.5-Math-1.5B-Instruct 数学模型进行解题。

1. 文件结构

压缩包解压后 (假设目录为 submission)，应包含以下核心文件和目录：

submission/
│
├── model/                  # (已包含) OCR 模型文件
├── model_math/             # (已包含) Math 模型文件
│
├── build_env.sh            # 脚本：一键安装 Python 依赖
├── run.sh                  # 脚本：一键运行代码
│
├── requirements.txt        # Python 依赖包列表
│
├── run.py                  # 主流水线控制脚本
├── ovis_ocr.py             # 阶段一：OCR 视觉模型代码
└── qwen_math.py            # 阶段二：Math 数学模型代码


2. 复现步骤 (核心)

请按照以下步骤操作，即可完成复现。

步骤 1: 上传与解压

将 submission.zip 压缩包上传到服务器的任意位置（例如 /root）。

解压缩文件:

unzip submission.zip


进入项目根目录:

cd submission


步骤 2: 构建环境 (安装依赖)

bash build_env.sh 进行环境配置；
pip install -r requirements.txt 安装依赖；


步骤 3: 运行代码

bash run.sh 启动模型推理。


脚本将开始运行，并在控制台打印 OCR 和 Math 阶段的进度条。

3. 结果说明

脚本运行成功后，会在您 run.sh 中指定的 OUTPUT_FILE 路径（例如 /path/to/your/output.jsonl）生成最终的答案文件。

该文件的格式为 jsonl，每一行包含原始的 image, tag 字段，以及模型生成的 step (空) 和 answer 字段。
