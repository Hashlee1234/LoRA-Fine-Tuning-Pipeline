import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
import os
import json
from typing import Optional


class LoRAPipeline:

    def __init__(
        self,
        base_model_path: str,
        output_dir: str = "./lora_output",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
    ):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules

        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def load_base_model(self):
        print(f"[LoRA] Loading base model from: {self.base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("[LoRA] Added pad_token = eos_token")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        total = sum(p.numel() for p in self.model.parameters())
        print(f"[LoRA] Base model loaded. Total params: {total:,}")

    def apply_lora(self):
        print(f"\n[LoRA] Applying LoRA adapters (r={self.lora_r}, alpha={self.lora_alpha})...")

        if self.target_modules is None:
            self.target_modules = self._detect_target_modules()

        print(f"[LoRA] Target modules: {self.target_modules}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )

        self.peft_model = get_peft_model(self.model, lora_config)

        trainable = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.peft_model.parameters())
        percent = 100 * trainable / total

        print(f"\n{'='*50}")
        print(f"  Trainable params : {trainable:,}")
        print(f"  Total params     : {total:,}")
        print(f"  Trainable %      : {percent:.4f}%")
        print(f"  Memory savings   : ~{100 - percent:.1f}% vs full fine-tuning")
        print(f"{'='*50}\n")

        return self.peft_model

    def _detect_target_modules(self) -> list:
        linear_layer_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_name = name.split(".")[-1]
                linear_layer_names.add(layer_name)

        print(f"[LoRA] Found linear layers: {linear_layer_names}")

        llama_style = {"q_proj", "v_proj"}
        if llama_style.issubset(linear_layer_names):
            return list(llama_style)

        if "c_attn" in linear_layer_names:
            return ["c_attn"]

        if "query_key_value" in linear_layer_names:
            return ["query_key_value"]

        return list(linear_layer_names)[:4]

    def prepare_dataset(self, texts: list[str], max_length: int = 512) -> Dataset:
        print(f"[LoRA] Tokenizing {len(texts)} training examples...")

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        raw_dataset = Dataset.from_dict({"text": texts})
        tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])

        print(f"[LoRA] Dataset ready: {len(tokenized)} samples")
        return tokenized

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 50,
        save_steps: int = 100,
        logging_steps: int = 10,
    ):
        if self.peft_model is None:
            raise RuntimeError("Call apply_lora() before train()")

        os.makedirs(self.output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            fp16=True,
            optim="adamw_torch",
            save_total_limit=2,
            load_best_model_at_end=False,
            report_to="none",
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        print(f"\n[LoRA] Starting training for {num_epochs} epochs...")
        print(f"[LoRA] Learning rate: {learning_rate} | Batch size: {batch_size}")
        print(f"[LoRA] Output dir: {self.output_dir}\n")

        trainer.train()

        print("\n[LoRA] Training complete!")
        return trainer

    def save_adapter(self):
        adapter_path = os.path.join(self.output_dir, "lora_adapter")
        print(f"\n[LoRA] Saving adapter to: {adapter_path}")

        self.peft_model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        total_size = sum(
            os.path.getsize(os.path.join(adapter_path, f))
            for f in os.listdir(adapter_path)
            if os.path.isfile(os.path.join(adapter_path, f))
        )
        print(f"[LoRA] Adapter saved! Size: {total_size / 1e6:.1f} MB")
        print(f"[LoRA] (Compare: base model was ~5500 MB)")

        return adapter_path

    def load_adapter(self, adapter_path: str):
        print(f"[LoRA] Loading adapter from: {adapter_path}")
        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        self.peft_model.eval()
        print("[LoRA] Adapter loaded successfully.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        model_to_use = self.peft_model if self.peft_model else self.model

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model_to_use.device)

        with torch.no_grad():
            output_ids = model_to_use.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


SAMPLE_TRAINING_DATA = [
    "Customer: I need 500 units of industrial valves for a chemical plant.\nVendor: Thank you for your inquiry. We can supply 500 SS304 industrial ball valves at ₹850/unit. Delivery: 15 working days. MOQ: 100 units. GST extra. Please share specs for custom requirements.",
    "Customer: Looking for bulk office chairs, need 200 units.\nVendor: We offer ergonomic mesh chairs at ₹3,200/unit for bulk of 200+. Includes 1-year warranty and free delivery within Mumbai. Customization available on armrests and color.",
    "Customer: Need 1000 kg of food-grade packaging material.\nVendor: We supply LDPE food-grade packaging bags at ₹120/kg for 1000kg order. BIS certified, FDA compliant. Lead time: 7 days. Payment: 50% advance, 50% on delivery.",
    "Customer: Require electrical cable installation services for new factory.\nVendor: We provide complete electrical cabling solutions. For factory projects, our rate is ₹45/sq.ft for full installation including materials. Site visit and detailed quote within 2 business days.",
    "Customer: Want to buy industrial pumps, around 50 units.\nVendor: Our centrifugal pumps for industrial use are priced at ₹12,500/unit for 50+ quantities. Available in 1HP to 10HP range. 2-year warranty. Installation support available at ₹2,000/unit extra.",
    "Customer: Need raw cotton for textile manufacturing, 5 tonnes.\nVendor: Long staple raw cotton available at ₹58/kg (₹58,000/tonne) for 5-tonne order. Origin: Gujarat. Moisture content 8%. Delivery FCA our warehouse. Bank LC preferred.",
    "Customer: Looking for IT hardware maintenance contract for 200 computers.\nVendor: Annual AMC for 200 desktops/laptops at ₹800/unit/year = ₹1.6L total. Includes quarterly servicing, software updates, onsite support within 4 hours. Contract period: 1 year renewable.",
    "Customer: Need catering services for 500-person corporate event.\nVendor: Corporate event catering for 500 pax: Veg menu ₹450/plate, Non-veg ₹600/plate. Includes setup, service staff, and cleanup. Advance booking of 2 weeks required. Customizable menu available.",
]


if __name__ == "__main__":
    import sys

    if "--dry-run" in sys.argv:
        print("="*60)
        print("LoRA Pipeline — Dry Run (structure test, no model loading)")
        print("="*60)

        print(f"\nSample training data ({len(SAMPLE_TRAINING_DATA)} examples):")
        for i, text in enumerate(SAMPLE_TRAINING_DATA[:2]):
            print(f"\n--- Example {i+1} ---")
            print(text[:200] + "..." if len(text) > 200 else text)

        print("\nLoRA config that would be used:")
        config = {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM",
        }
        for k, v in config.items():
            print(f"  {k}: {v}")

        print("\nTraining args that would be used:")
        args = {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": "2e-4",
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 16,
            "fp16": True,
        }
        for k, v in args.items():
            print(f"  {k}: {v}")

        print("\n✓ Dry run complete. Run with a real model path for actual training.")

    else:
        model_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\models\phi-2"

        pipeline = LoRAPipeline(
            base_model_path=model_path,
            output_dir="./lora_output",
            lora_r=16,
            lora_alpha=32,
        )

        pipeline.load_base_model()
        pipeline.apply_lora()

        dataset = pipeline.prepare_dataset(SAMPLE_TRAINING_DATA)
        pipeline.train(dataset, num_epochs=3, batch_size=2)

        adapter_path = pipeline.save_adapter()

        print("\n--- Testing fine-tuned model ---")
        prompt = "Customer: I need 300 units of stainless steel pipes.\nVendor:"
        response = pipeline.generate(prompt)
        print(f"Prompt : {prompt}")
        print(f"Response: {response}")
