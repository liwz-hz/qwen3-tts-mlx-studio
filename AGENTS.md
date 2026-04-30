# Qwen3-TTS MLX Studio — Agent 操作指南

## 项目概述

基于 Apple MLX 框架的本地 AI 文本转语音（TTS）工具，使用 ModelScope（魔搭社区）作为模型源，运行在 conda `audio` 环境中。

## 环境要求

- macOS + Apple Silicon (M1/M2/M3/M4)
- Miniconda / Anaconda
- Python 3.10-3.13
- ffmpeg（音频编码）

## ModelScope 与 HuggingFace

ModelScope 是 HuggingFace 的平行社区，模型仓库结构和命名基本一致。本项目使用 ModelScope 替代 HuggingFace 作为模型下载源。

- HuggingFace 模型路径：`~/.cache/huggingface/hub/models--mlx-community--...`
- ModelScope 模型路径：`~/.cache/modelscope/hub/models/mlx-community/...`

两者模型格式完全相同（均为 MLX 格式的 `.safetensors` 文件），可直接互换使用。

## 模型下载

### 可用模型

| 模型变体 | ModelScope 路径 | 说明 |
|----------|----------------|------|
| CustomVoice | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-{bf16\|8bit\|6bit\|4bit}` | 自定义音色 |
| VoiceDesign | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-{bf16\|8bit\|6bit\|4bit}` | 音色设计 |
| Base | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-{bf16\|8bit\|6bit\|4bit}` | 音色克隆基础模型 |

### 下载方式

**方法 1：浏览器下载**

访问 ModelScope 页面点击直接下载：
- https://www.modelscope.cn/models/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
- https://www.modelscope.cn/models/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit
- https://www.modelscope.cn/models/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit

**方法 2：CLI 下载**

```bash
pip install modelscope
modelscope download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
```

**方法 3：Python 代码下载**

```python
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
```

### 模型缓存路径

模型默认下载到：
```
~/.cache/modelscope/hub/models/mlx-community/
```

目录结构示例：
```
~/.cache/modelscope/hub/models/mlx-community/
├── Qwen3-TTS-12Hz-1.7B-Base-bf16/
├── Qwen3-TTS-12Hz-1.7B-Base-8bit/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16/
└── Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit/
```

每个模型目录包含：
- `config.json` — 模型配置
- `model.safetensors` — 模型权重
- `tokenizer_config.json`, `vocab.json` — 分词器
- `speech_tokenizer/` — 语音分词器
- 等其他必要文件

## 安装与运行

### 首次安装

```bash
# 1. 克隆项目
git clone <repo-url>
cd qwen3-tts-mlx-studio

# 2. 运行安装脚本（自动创建 conda 环境 + 安装依赖）
./install.sh

# 3. 下载模型（可选，首次使用时会自动从本地缓存加载）
modelscope download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
```

### 启动应用

```bash
./run.sh
```

启动后浏览器自动打开 `http://localhost:7860`

### 卸载

```bash
./uninstall.sh
```

删除 conda `audio` 环境和 ModelScope 缓存中的模型。

## 模型加载机制

### 核心流程

1. `engine.py` 中的 `get_repo_id()` 方法构建本地路径：
   ```
   ~/.cache/modelscope/hub/models/mlx-community/Qwen3-TTS-12Hz-1.7B-{Variant}-{Quant}
   ```

2. `load_model(repo_id)` 接受本地路径作为参数，直接从本地加载 MLX 模型

3. ASR 模型（`Qwen3-ASR-1.7B-8bit`）和 DeepFilterNet 模型按需加载，首次使用时自动下载

### 缓存检测

`app.py` 中的 `_is_model_cached()` 函数通过 `os.path.isdir(repo_id)` 检查本地路径是否存在且非空。

### 模型删除

UI 中的 "Delete Downloaded Models" 按钮会遍历 `~/.cache/modelscope/hub/models/mlx-community/` 目录，删除所有以 `Qwen3-TTS-12Hz-` 开头的模型目录。

## 配置文件 (`config.py`)

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `MODELSCOPE_CACHE_ROOT` | `~/.cache/modelscope/hub/models/mlx-community` | 模型本地缓存路径 |
| `REPO_TEMPLATE` | `mlx-community/Qwen3-TTS-12Hz-{size}-{variant}-{quant}` | 模型命名模板 |
| `DEFAULT_QUANTIZATION` | `bf16` | 默认量化精度（bf16/8bit/6bit/4bit） |
| `ASR_REPO_ID` | `mlx-community/Qwen3-ASR-1.7B-8bit` | ASR 模型 |
| `DEEPFILTER_REPO` | `mlx-community/DeepFilterNet-mlx` | 降噪模型 |

## Conda 环境

- 环境名：`audio`
- Python 版本：3.12（推荐）
- 依赖：`requirements.txt` 中的所有包
- 激活方式：`conda activate audio`

## 量化版本说明

| 量化 | 模型大小 | 推理速度 | 音质 | 推荐场景 |
|------|---------|---------|------|---------|
| bf16 | ~3.4GB | 最快 | 最佳 | 内存充足时 |
| 8bit | ~1.7GB | 快 | 很好 | 日常使用（推荐）|
| 6bit | ~1.3GB | 中等 | 好 | 内存受限时 |
| 4bit | ~850MB | 较慢 | 可接受 | 最低内存需求 |

## 故障排查

### 模型未找到

检查缓存目录是否存在：
```bash
ls ~/.cache/modelscope/hub/models/mlx-community/
```

如为空，需手动下载模型。

### Conda 环境问题

```bash
# 查看环境列表
conda env list

# 手动激活
conda activate audio

# 重新安装依赖
conda activate audio
pip install -r requirements.txt
```

### 模型加载失败

确认 `mlx-audio` 版本正确：
```bash
conda activate audio
pip show mlx-audio
```

需要 `>= 0.4.2`。
