<div align="center">

# 🖼️ IMG-GC

**Image-to-Text & Image-to-Image generation web app**
**图像描述生成与图生图 Web 应用**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?style=flat-square&logo=bootstrap&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Diffusers%20%7C%20Transformers-FFD21E?style=flat-square)

</div>

---

## 📖 Overview · 简介

**English** — A Flask + Bootstrap website with two generative features:
- **Image → Text**: BLIP-2 (`Salesforce/blip2-opt-2.7b`) fine-tuned with **LoRA** (PEFT) for image captioning, with optional text-to-speech (gTTS).
- **Image → Image**: **Stable Diffusion** img2img (`StableDiffusionImg2ImgPipeline`).

**中文** — 基于 Flask 与 Bootstrap 的网站，提供两种生成功能：
- **图生文**：使用 BLIP-2（`Salesforce/blip2-opt-2.7b`），通过 **LoRA**（PEFT）微调后生成图像描述，可选 gTTS 语音播报。
- **图生图**：基于 **Stable Diffusion** img2img 管线。

## 🚀 Quick Start · 快速开始

```bash
pip install -r requirements.txt
python myweb/app.py
```

Open the printed local address in a browser. A CUDA GPU is recommended. 浏览器访问终端输出的本地地址即可，建议使用 CUDA GPU。

## 📁 Structure · 目录结构

```text
IMG-GC
├── myweb/
│   ├── app.py          # Flask routes: /, /img2text, /img2img
│   └── templates/      # Jinja2 + Bootstrap pages
├── utils/
│   ├── models.py       # BLIP-2 loading & LoRA fine-tuning
│   ├── get_input.py    # image loading / preprocessing
│   ├── text2voice.py   # gTTS text-to-speech
│   └── *.ipynb         # training & SD experiments
├── flask_bootstrap/    # vendored Bootstrap-Flask
└── requirements.txt
```
