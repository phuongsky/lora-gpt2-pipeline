project/
├── config.yaml                  # ✅ Cấu hình pipeline (dataset, model, training, LoRA)
├── train.py                     # ✅ Script train chính
├── infer_api.py                 # ✅ FastAPI Inference Server
├── utils.py                     # ✅ Helper (load config, tokenizer, etc.)
└── requirements.txt             # ✅ Các thư viện cần thiết
# Huấn luyện
python train.py

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
