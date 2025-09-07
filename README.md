project/
├── config.yaml                  # ✅ Cấu hình pipeline (dataset, model, training, LoRA)
├── train.py                     # ✅ Script train chính
├── infer_api.py                 # ✅ FastAPI Inference Server
├── utils.py                     # ✅ Helper (load config, tokenizer, etc.)
└── requirements.txt             # ✅ Các thư viện cần thiết
# Huấn luyện
python train.py
Chạy UI:
python app_ui.py
# Chạy inference API
uvicorn infer_api:app --reload --port 8000
POST http://localhost:8000/generate
{
  "prompt": "Once upon a time",
  "max_new_tokens": 50,
  "temperature": 0.8
}
lora-nlp-project/
├── config.yaml                   # YAML cấu hình toàn bộ pipeline
├── train.py                      # Training script chính (multi-GPU)
├── utils.py                      # Helper: load config, tokenizer, model
├── infer_api.py                  # FastAPI inference server
├── app_ui.py                     # Gradio / Streamlit UI
├── Dockerfile                    # Docker hóa inference API
├── requirements.txt              # Requirements cho training + inference
├── README.md                     # Hướng dẫn
└── .gitignore
Build Docker image:
docker build -t lora-infer .
Chạy inference API:
docker run -p 8000:8000 lora-infer
⚡ 4. Huấn luyện Multi-GPU + DeepSpeed + WandB
Chạy huấn luyện multi-GPU:
accelerate launch --multi_gpu train.py
Hoặc với deepspeed:
deepspeed train.py

# 🤖 LoRA NLP Pipeline with Quantization, Gradio UI & Inference API

This project provides a complete end-to-end solution for:
- Fine-tuning `DistilGPT2` with 🧠 **LoRA** (via `peft`)
- Efficient training with 🔥 **DeepSpeed + quantization (4-bit/8-bit)**
- Deployment-ready **FastAPI** server + **Gradio UI**
- YAML-based config for reproducibility
- Optional W&B logging

## 🚀 Features

- ✅ Train with LoRA & HuggingFace PEFT
- ✅ Multi-dataset & multi-GPU training
- ✅ Quantized inference with bitsandbytes
- ✅ Push adapter to Hugging Face Hub
- ✅ Docker container for API deployment
- ✅ Gradio UI for demo

## 🛠️ Quickstart

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Train model
python train.py

# 3. Run API
uvicorn infer_api:app --port 8000

# 4. Run UI
python app_ui.py
----------------------------

# LoRA GPT‑2 Pipeline: Train, Quantize, Serve, UI & Docker

##  Features
- Fine‑tune DistilGPT‑2 with LoRA + 4‑bit Quantization
- Multi‑GPU training with DeepSpeed
- Logging via Weights & Biases (W&B)
- Inference via FastAPI + Gradio UI
- Docker container for easy deployment
- YAML‑based configuration for reproducibility
- Push adapter + tokenizer to Hugging Face Hub

##  Setup & Usage

###  Clone & install
```bash
git clone https://github.com/your-username/lora-gpt2-pipeline.git
cd lora-gpt2-pipeline
pip install -r requirements.txt
