# 西游记微调Gemma 2大模型项目

本项目旨在利用西游记对Gemma 2大语言模型进行微调,以提升模型在中文文学领域的表现。

## 训练环境

本项目的训练环境为:
- Windows的WSL2 (Windows Subsystem for Linux 2)
或
- Ubuntu操作系统

请确保您的系统满足以上环境之一,以便顺利进行模型训练。

## 环境配置

1. 创建Conda环境:

```bash
conda create -n unsloth python=3.10
conda activate unsloth
```

2. 安装必要的依赖:

```bash
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps --upgrade "flash-attn>=2.6.3"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install einops
pip install protobuf==3.20.3
```

3. 模型下载
需要去huggingface下载gemma-2-2b-it-bnb-4bit模型，由于GitHub文件上传限制，无法进行上传。

## 模型运行
执行命令
```bash
python train.py
```

## 模型保存

训练完成后,模型将自动保存在项目根目录下的`model`文件夹中。

## 项目说明

本项目使用西游记作为训练数据,演示对Gemma2大语言模型进行微调的过程
