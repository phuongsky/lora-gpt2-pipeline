from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI()

# Load model + tokenizer
repo_id = "yourusername/lora-distilgpt2"
base_model = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, repo_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define input format
class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7

@app.post("/generate")
def generate_text(req: GenRequest):
    output = pipe(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return {"output": output[0]["generated_text"]}
