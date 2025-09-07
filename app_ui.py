gr.Interface(
    fn=generate,
    inputs=[...],
    outputs=...,
    title="🚀 LoRA Inference",
    description="Fine-tuned DistilGPT2 + LoRA"
).launch()
