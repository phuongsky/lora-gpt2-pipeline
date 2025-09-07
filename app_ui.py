import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
repo_id = "phuongsky/lora-distilgpt2"
base_model = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, repo_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate(prompt, max_tokens, temp):
    output = pipe(prompt, max_new_tokens=max_tokens, temperature=temp)
    return output[0]["generated_text"]

gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=2, label="Prompt"),
        gr.Slider(10, 200, value=50, step=10, label="Max Tokens"),
        gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="🚀 LoRA Inference",
    description="Inference from fine-tuned DistilGPT2 + LoRA"
).launch()
