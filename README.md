# LoRA Fine-Tuning Pipeline

I built this after trying to understand why fine-tuning a language model costs so much money. Turns out, you don't have to update all 2.7 billion weights — you can freeze almost everything and inject tiny trainable matrices into just the attention layers. That's LoRA. The adapter you end up with is around 20MB. The base model is 5.5GB. I wanted to see that with my own eyes, so I built a pipeline that shows you the exact numbers.

The context I built this for: [Janooma](https://janooma.com) is building an AI marketplace where vendors need specialized models. LoRA is exactly how you'd adapt a base model for each vendor's domain without spinning up a new training run from scratch every time.

---

## What it does

You give it a base model and a set of training examples. It injects LoRA adapters into the attention layers, trains only those (about 0.06% of total parameters), saves the adapter, then lets you compare the fine-tuned outputs against the base model using ROUGE scores.

There's a Streamlit UI where you can configure everything — rank, alpha, dropout, learning rate — and see exactly how many parameters become trainable as you change the rank. The math is right there: trainable% = (2 × r × d) / total_params.

---

## Files

```
lora_core.py      — the whole pipeline: load → inject LoRA → train → save → generate
evaluator.py      — ROUGE-1, ROUGE-2, ROUGE-L comparison between base and fine-tuned
app.py            — Streamlit UI wiring the above into something you can click around in
requirements.txt  — dependencies
```

---

## How to run it

```bash
pip install -r requirements.txt

# Test the structure without loading any model
python lora_core.py --dry-run
python evaluator.py --dry-run

# Launch the UI
streamlit run app.py
```

For actual training, use Google Colab — free T4 GPU, and the pipeline loads from HuggingFace directly. The Train tab in the UI has copy-pasteable Colab code for the exact config you've set in the sliders.

---

## The LoRA math, briefly

Standard fine-tuning updates W directly. LoRA freezes W and adds two small matrices A and B (where r << d), so the update is W + BA. During training, only A and B are updated. After training, you can either keep them separate (adapter mode) or merge them back into W permanently.

The rank r controls how much expressiveness you're buying. In this pipeline, r=16 gives you about 13 million trainable parameters out of 2.7 billion — that's 0.06%. The adapter file is ~20MB. The base model is 5.5GB. You load the base model once and swap adapters depending on which vendor domain you need.

---

## What the evaluation tab shows

After training, run:

```bash
python evaluator.py /path/to/base/model ./lora_output/lora_adapter
```

It generates responses from both the base model and the fine-tuned model for the same prompts, scores them against reference outputs using ROUGE, and saves a full JSON report. The UI loads this report and shows you the delta for ROUGE-1 and ROUGE-L side by side.

The training data I used is vendor-quote style — customer inquiry → structured vendor response with pricing, delivery, and payment terms. That's the domain Janooma's marketplace needs.

---

## What I learned building this

The thing that surprised me most: the target_modules detection. You can't just hardcode `["q_proj", "v_proj"]` — different model families name their attention layers differently. Phi-2 uses `q_proj`/`v_proj`, GPT-2 uses `c_attn`, older models use `query_key_value`. The `_detect_target_modules()` method in lora_core.py handles this by scanning the actual layer names at runtime.

Also: `lora_alpha / lora_r` is the effective scaling factor, not lora_alpha alone. I was confused about this for a while. Setting alpha=32 and r=16 gives you scale=2.0, which is the standard recommendation. The UI shows you this ratio live as you move the sliders.

---

## Requirements

- Python 3.10+
- CUDA GPU recommended for training (CPU works but is very slow)
- ~6GB VRAM for phi-2 in float16

For inference and evaluation on CPU, it works fine — just slower.

---

## Model

This was built and tested with microsoft/phi-2. Download from [HuggingFace](https://huggingface.co/microsoft/phi-2) and point the model path in the UI to your local folder, or use the HuggingFace model name directly if you have internet access during the run.

---

## Related projects

- [LLM Architecture Inspector](https://github.com/Hashlee1234/llm-architecture-inspector) — visualizes the model structure that LoRA targets
- [Model Merger](https://github.com/Hashlee1234/Model-Merger) — SLERP/TIES/DARE weight merging, the alternative to fine-tuning
- [Conversational RAG Agent](https://github.com/Hashlee1234/conversational-rag-agent) — uses a fine-tuned model as the generation layer

Happy to answer questions if anything in the code is unclear.
