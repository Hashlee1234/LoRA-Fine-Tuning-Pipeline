# LoRA Fine-Tuning Pipeline

I built this after trying to understand why fine-tuning a language model costs so much money. Turns out, you don't have to update all 2.7 billion weights — you can freeze almost everything and inject tiny trainable matrices into just the attention layers. That's LoRA. The adapter you end up with is around 20MB. The base model is 5.5GB. I wanted to see that with my own eyes, so I built a pipeline that shows you the exact numbers.

## What it does

You give it a base model and a set of training examples. It injects LoRA adapters into the attention layers, trains only those (about 0.06% of total parameters), saves the adapter, then lets you compare the fine-tuned outputs against the base model using ROUGE scores.

There's a Streamlit UI where you can configure everything — rank, alpha, dropout, learning rate — and see exactly how many parameters become trainable as you change the rank. The math is right there: trainable% = (2 × r × d) / total_params.


## Files

```
lora_core.py      — the whole pipeline: load → inject LoRA → train → save → generate
evaluator.py      — ROUGE-1, ROUGE-2, ROUGE-L comparison between base and fine-tuned
app.py            — Streamlit UI wiring the above into something you can click around in
requirements.txt  — dependencies
```


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


## What the evaluation tab shows

After training, run:

```bash
python evaluator.py /path/to/base/model ./lora_output/lora_adapter
```

It generates responses from both the base model and the fine-tuned model for the same prompts, scores them against reference outputs using ROUGE, and saves a full JSON report. The UI loads this report and shows you the delta for ROUGE-1 and ROUGE-L side by side.


## Model

This was built and tested with microsoft/phi-2. Download from [HuggingFace](https://huggingface.co/microsoft/phi-2) and point the model path in the UI to your local folder, or use the HuggingFace model name directly if you have internet access during the run.


