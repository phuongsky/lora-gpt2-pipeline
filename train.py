import os
import torch
import yaml
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, set_seed
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel
from huggingface_hub import login
from utils import load_config, get_model_tokenizer

# ============================
# 1. Load config
# ============================
cfg = load_config("config.yaml")
set_seed(cfg['training']['seed'])

# ============================
# 2. Load model & tokenizer
# ============================
model, tokenizer = get_model_tokenizer(cfg)

# ============================
# 3. Apply LoRA
# ============================
lora_cfg = cfg['lora']
lora_config = LoraConfig(
    r=lora_cfg['r'],
    lora_alpha=lora_cfg['lora_alpha'],
    target_modules=lora_cfg['target_modules'],
    lora_dropout=lora_cfg['dropout'],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================
# 4. Load dataset
# ============================
data_cfg = cfg['dataset']
train_dataset = load_dataset(data_cfg['train_dataset'])['train']
valid_dataset = load_dataset(data_cfg['valid_dataset'])['train']  # Nếu không có split validation

def tokenize_fn(examples):
    return tokenizer(
        examples[data_cfg['text_column']],
        truncation=True,
        padding="max_length",
        max_length=data_cfg['max_length']
    )

train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=[data_cfg['text_column']])
valid_dataset = valid_dataset.map(tokenize_fn, batched=True, remove_columns=[data_cfg['text_column']])

# ============================
# 5. Training
# ============================
args = TrainingArguments(
    output_dir=cfg['training']['output_dir'],
    per_device_train_batch_size=int(cfg['training']['batch_size']),
    num_train_epochs=float(cfg['training']['epochs']),
    learning_rate=float(cfg['training']['learning_rate']),
    logging_steps=int(cfg['training']['logging_steps']),
    save_steps=int(cfg['training']['save_steps']),
    eval_strategy="no",
    save_total_limit=1,
    fp16=bool(cfg['training']['fp16']),
    deepspeed=cfg['training'].get('deepspeed', None),
    report_to=cfg['training'].get('report_to', "none"),
    run_name=cfg['training'].get('run_name', None)
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()

# ============================
# 6. Save + Push
# ============================
model.save_pretrained(cfg['training']['output_dir'])
tokenizer.save_pretrained(cfg['training']['output_dir'])

if cfg['hf_hub']['push_to_hub']:
    token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    if token:
        login(token=token)
    else:
        print("Hugging Face token not set in environment; skipping login to avoid interactive prompt.")
    model.push_to_hub(cfg['hf_hub']['repo_id'], token=token)
    tokenizer.push_to_hub(cfg['hf_hub']['repo_id'], token=token)
