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
