import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
import json
import os
from typing import Optional


class LoRAEvaluator:

    def __init__(
        self,
        base_model_path: str,
        adapter_path: Optional[str] = None,
    ):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path

        self.base_model = None
        self.finetuned_model = None
        self.tokenizer = None

        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def load_models(self):
        print("[Evaluator] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[Evaluator] Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.base_model.eval()

        if self.adapter_path and os.path.exists(self.adapter_path):
            print("[Evaluator] Loading fine-tuned model (base + LoRA adapter)...")
            self.finetuned_model = PeftModel.from_pretrained(
                self.base_model,
                self.adapter_path
            )
            self.finetuned_model.eval()
            print("[Evaluator] Both models loaded.")
        else:
            print("[Evaluator] No adapter path — only base model loaded.")

    def generate_response(self, model, prompt: str, max_new_tokens: int = 150) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def compute_rouge(self, hypothesis: str, reference: str) -> dict:
        scores = self.rouge.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }

    def evaluate(
        self,
        test_prompts: list[dict],
        output_path: str = "./evaluation_report.json"
    ) -> list[dict]:
        results = []

        for i, item in enumerate(test_prompts):
            prompt = item["prompt"]
            reference = item.get("reference", "")

            print(f"\n[Evaluator] Evaluating prompt {i+1}/{len(test_prompts)}...")
            print(f"  Prompt: {prompt[:80]}...")

            result = {
                "prompt": prompt,
                "reference": reference,
                "base_output": "",
                "finetuned_output": "",
                "base_rouge": {},
                "finetuned_rouge": {},
                "improvement": {},
            }

            print("  Generating base model response...")
            result["base_output"] = self.generate_response(self.base_model, prompt)

            if self.finetuned_model:
                print("  Generating fine-tuned model response...")
                result["finetuned_output"] = self.generate_response(self.finetuned_model, prompt)

            if reference:
                result["base_rouge"] = self.compute_rouge(result["base_output"], reference)

                if result["finetuned_output"]:
                    result["finetuned_rouge"] = self.compute_rouge(result["finetuned_output"], reference)

                    result["improvement"] = {
                        metric: round(result["finetuned_rouge"].get(metric, 0) - result["base_rouge"].get(metric, 0), 4)
                        for metric in ["rouge1", "rouge2", "rougeL"]
                    }

            results.append(result)

            print(f"  Base output:      {result['base_output'][:100]}...")
            if result["finetuned_output"]:
                print(f"  Fine-tuned output: {result['finetuned_output'][:100]}...")
            if result["base_rouge"]:
                print(f"  Base ROUGE-L:     {result['base_rouge']['rougeL']}")
            if result["finetuned_rouge"]:
                print(f"  Fine-tuned ROUGE-L: {result['finetuned_rouge']['rougeL']}")
                improvement = result["improvement"].get("rougeL", 0)
                direction = "↑" if improvement >= 0 else "↓"
                print(f"  Improvement:      {direction} {abs(improvement):.4f}")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        self._print_summary(results)

        print(f"\n[Evaluator] Full report saved to: {output_path}")
        return results

    def _print_summary(self, results: list[dict]):
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        if not any(r["base_rouge"] for r in results):
            print("No reference texts provided — ROUGE scores not computed.")
            print(f"Generated responses for {len(results)} prompts.")
            return

        metrics = ["rouge1", "rouge2", "rougeL"]

        base_avg = {m: sum(r["base_rouge"].get(m, 0) for r in results) / len(results) for m in metrics}
        ft_avg = {m: sum(r["finetuned_rouge"].get(m, 0) for r in results if r["finetuned_rouge"]) / max(1, sum(1 for r in results if r["finetuned_rouge"])) for m in metrics}

        print(f"\n{'Metric':<12} {'Base Model':>12} {'Fine-Tuned':>12} {'Delta':>10}")
        print("-" * 50)
        for m in metrics:
            delta = ft_avg[m] - base_avg[m]
            direction = "↑" if delta >= 0 else "↓"
            print(f"{m:<12} {base_avg[m]:>12.4f} {ft_avg[m]:>12.4f} {direction}{abs(delta):>8.4f}")

        print("="*60)


TEST_PROMPTS = [
    {
        "prompt": "Customer: I need 100 units of industrial motors.\nVendor:",
        "reference": "We supply 3-phase induction motors at ₹8,500/unit for 100 qty. Available in 1HP-50HP range. ISI marked. Delivery 10 working days. 1-year warranty included.",
    },
    {
        "prompt": "Customer: Looking for bulk packaging solutions for food products.\nVendor:",
        "reference": "We offer food-grade LDPE packaging at competitive rates. For bulk orders, pricing starts ₹95/kg. BIS and FDA certified. Minimum order 500kg. Customized sizes available.",
    },
    {
        "prompt": "Customer: Need annual maintenance contract for office equipment.\nVendor:",
        "reference": "Our AMC covers all office equipment at ₹1,200/unit/year. Includes quarterly servicing, emergency support within 6 hours, and all spare parts. 1-year contract renewable.",
    },
]


if __name__ == "__main__":
    import sys

    if "--dry-run" in sys.argv:
        print("="*60)
        print("LoRA Evaluator — Dry Run")
        print("="*60)

        print(f"\nTest prompts ({len(TEST_PROMPTS)}):")
        for i, p in enumerate(TEST_PROMPTS):
            print(f"\n[{i+1}] Prompt   : {p['prompt']}")
            print(f"     Reference: {p['reference'][:80]}...")

        print("\n✓ Evaluator structure is correct.")
        print("  Run with a model path to do actual generation + ROUGE scoring.")
    else:
        model_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\models\phi-2"
        adapter_path = sys.argv[2] if len(sys.argv) > 2 else "./lora_output/lora_adapter"

        evaluator = LoRAEvaluator(
            base_model_path=model_path,
            adapter_path=adapter_path,
        )
        evaluator.load_models()
        evaluator.evaluate(TEST_PROMPTS, output_path="./lora_output/evaluation_report.json")
