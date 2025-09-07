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
