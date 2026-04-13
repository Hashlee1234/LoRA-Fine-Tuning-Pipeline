import streamlit as st
import json
import os

st.set_page_config(
    page_title="LoRA Fine-Tuning Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .stage-badge {
        background: #45475a;
        color: #cdd6f4;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .improvement-pos { color: #a6e3a1; font-weight: bold; }
    .improvement-neg { color: #f38ba8; font-weight: bold; }
    code { background: #313244; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ LoRA Configuration")
    st.divider()

    st.subheader("Model")
    model_path = st.text_input(
        "Base model path",
        value=r"D:\models\phi-2",
        help="Local folder path or HuggingFace model name"
    )
    output_dir = st.text_input("Output directory", value="./lora_output")

    st.divider()
    st.subheader("LoRA Hyperparameters")

    lora_r = st.slider(
        "Rank (r)",
        min_value=4, max_value=64, value=16, step=4,
        help="Higher rank = more capacity = more memory"
    )
    lora_alpha = st.number_input(
        "Alpha",
        value=32,
        help="Scaling = alpha/r. Usually set to 2× rank."
    )
    lora_dropout = st.slider(
        "Dropout",
        min_value=0.0, max_value=0.2, value=0.05, step=0.01,
        help="Regularization to prevent overfitting"
    )

    effective_scale = lora_alpha / lora_r
    st.caption(f"Effective scale: **{effective_scale:.2f}** (alpha/r)")

    st.divider()
    st.subheader("Training Hyperparameters")

    num_epochs = st.slider("Epochs", 1, 10, 3)
    batch_size = st.selectbox("Batch size", [1, 2, 4, 8], index=1)
    learning_rate = st.select_slider(
        "Learning rate",
        options=["5e-5", "1e-4", "2e-4", "3e-4", "5e-4"],
        value="2e-4"
    )
    max_length = st.slider("Max token length", 128, 1024, 512, step=64)

    st.divider()

    param_estimate = lora_r * 2 * 2048
    st.caption(f"Estimated trainable params: ~{param_estimate:,}")
    st.caption(f"~{param_estimate / 2.7e9 * 100:.3f}% of phi-2 (2.7B)")

st.title("LoRA Fine-Tuning Pipeline")
st.caption("Train, evaluate, and test a LoRA-adapted language model — no full GPU farm needed")

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Training Data",
    "🚀 Train",
    "📊 Evaluate",
    "💬 Inference"
])

with tab1:
    st.subheader("Training Data")
    st.caption("Each entry is one training example. The model learns to continue text in this style.")

    default_data = [
        "Customer: I need 500 units of industrial valves for a chemical plant.\nVendor: Thank you for your inquiry. We can supply 500 SS304 industrial ball valves at ₹850/unit. Delivery: 15 working days. MOQ: 100 units. GST extra.",
        "Customer: Looking for bulk office chairs, need 200 units.\nVendor: We offer ergonomic mesh chairs at ₹3,200/unit for bulk of 200+. Includes 1-year warranty and free delivery within Mumbai.",
        "Customer: Need 1000 kg of food-grade packaging material.\nVendor: We supply LDPE food-grade packaging bags at ₹120/kg for 1000kg order. BIS certified, FDA compliant. Lead time: 7 days.",
        "Customer: Require electrical cable installation services for new factory.\nVendor: We provide complete electrical cabling solutions. For factory projects, our rate is ₹45/sq.ft including materials.",
        "Customer: Want to buy industrial pumps, around 50 units.\nVendor: Our centrifugal pumps are priced at ₹12,500/unit for 50+ quantities. Available in 1HP to 10HP. 2-year warranty included.",
    ]

    if "training_data" not in st.session_state:
        st.session_state.training_data = default_data.copy()

    col1, col2 = st.columns([3, 1])
    with col1:
        new_example = st.text_area(
            "Add new training example",
            placeholder="Customer: [inquiry]\nVendor: [ideal response]",
            height=100
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("➕ Add Example", use_container_width=True):
            if new_example.strip():
                st.session_state.training_data.append(new_example.strip())
                st.success("Example added!")
                st.rerun()

    st.divider()
    st.write(f"**{len(st.session_state.training_data)} training examples:**")

    for i, example in enumerate(st.session_state.training_data):
        with st.expander(f"Example {i+1}: {example[:60]}..."):
            st.code(example, language=None)
            if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                st.session_state.training_data.pop(i)
                st.rerun()

    st.divider()
    training_json = json.dumps(st.session_state.training_data, indent=2, ensure_ascii=False)
    st.download_button(
        "💾 Download Training Data (JSON)",
        data=training_json,
        file_name="training_data.json",
        mime="application/json"
    )

with tab2:
    st.subheader("Launch Fine-Tuning")

    st.write("**Pre-flight checklist:**")

    checks = {
        f"Model path set: `{model_path}`": bool(model_path),
        f"Training data: {len(st.session_state.training_data)} examples": len(st.session_state.training_data) > 0,
        "LoRA config: valid": lora_r > 0 and lora_alpha > 0,
    }

    all_clear = True
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        st.write(f"{icon} {check}")
        if not status:
            all_clear = False

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("LoRA Rank", lora_r)
    col2.metric("Epochs", num_epochs)
    col3.metric("Batch Size", batch_size)

    col4, col5, col6 = st.columns(3)
    col4.metric("Learning Rate", learning_rate)
    col5.metric("Alpha", lora_alpha)
    col6.metric("Max Length", max_length)

    st.divider()

    st.write("**Run this command in your terminal:**")
    st.code(
        f"python lora_core.py {model_path}",
        language="bash"
    )

    st.write("**For Colab (free GPU):**")
    colab_code = f"""
# In Google Colab:
!pip install transformers peft datasets accelerate rouge-score

from lora_core import LoRAPipeline, SAMPLE_TRAINING_DATA

pipeline = LoRAPipeline(
    base_model_path="microsoft/phi-2",
    output_dir="/content/lora_output",
    lora_r={lora_r},
    lora_alpha={lora_alpha},
    lora_dropout={lora_dropout},
)

pipeline.load_base_model()
pipeline.apply_lora()

dataset = pipeline.prepare_dataset(SAMPLE_TRAINING_DATA, max_length={max_length})
pipeline.train(dataset, num_epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate})

adapter_path = pipeline.save_adapter()
print(f"Adapter saved to: {{adapter_path}}")
"""
    st.code(colab_code, language="python")

    st.info(f"""
    **Estimated training time:**
    - Google Colab T4 GPU: ~{num_epochs * len(st.session_state.training_data) // 4} minutes
    - Local CPU only: ~{num_epochs * len(st.session_state.training_data) * 2} minutes (very slow, not recommended)
    
    **Adapter size after training:** ~15-25 MB (tiny!)
    """)

with tab3:
    st.subheader("Evaluation Results")

    adapter_path = os.path.join(output_dir, "lora_adapter")
    report_path = os.path.join(output_dir, "evaluation_report.json")

    if os.path.exists(report_path):
        with open(report_path) as f:
            results = json.load(f)

        st.success(f"Loaded evaluation report: {len(results)} test prompts")

        base_r1 = sum(r["base_rouge"].get("rouge1", 0) for r in results if r["base_rouge"]) / max(1, len(results))
        ft_r1   = sum(r["finetuned_rouge"].get("rouge1", 0) for r in results if r["finetuned_rouge"]) / max(1, len(results))
        base_rl = sum(r["base_rouge"].get("rougeL", 0) for r in results if r["base_rouge"]) / max(1, len(results))
        ft_rl   = sum(r["finetuned_rouge"].get("rougeL", 0) for r in results if r["finetuned_rouge"]) / max(1, len(results))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Base ROUGE-1", f"{base_r1:.4f}")
        col2.metric("Fine-tuned ROUGE-1", f"{ft_r1:.4f}", delta=f"{ft_r1 - base_r1:+.4f}")
        col3.metric("Base ROUGE-L", f"{base_rl:.4f}")
        col4.metric("Fine-tuned ROUGE-L", f"{ft_rl:.4f}", delta=f"{ft_rl - base_rl:+.4f}")

        st.divider()

        for i, r in enumerate(results):
            with st.expander(f"Prompt {i+1}: {r['prompt'][:70]}..."):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.write("**Base model output:**")
                    st.info(r["base_output"] or "—")
                    if r["base_rouge"]:
                        st.caption(f"ROUGE-1: {r['base_rouge']['rouge1']} | ROUGE-L: {r['base_rouge']['rougeL']}")

                with col_b:
                    st.write("**Fine-tuned output:**")
                    st.success(r["finetuned_output"] or "—")
                    if r["finetuned_rouge"]:
                        st.caption(f"ROUGE-1: {r['finetuned_rouge']['rouge1']} | ROUGE-L: {r['finetuned_rouge']['rougeL']}")

                if r.get("reference"):
                    st.write("**Reference (ideal output):**")
                    st.write(r["reference"])

        st.download_button(
            "📥 Download Full Report (JSON)",
            data=json.dumps(results, indent=2),
            file_name="evaluation_report.json",
            mime="application/json"
        )
    else:
        st.warning("No evaluation report found yet.")
        st.write("After training, run:")
        st.code(f"python evaluator.py {model_path} {adapter_path}", language="bash")
        st.write("Then refresh this page.")

with tab4:
    st.subheader("Test the Fine-Tuned Model")

    st.write("**Sample prompts to try:**")
    sample_prompts = [
        "Customer: I need 300 units of stainless steel pipes.\nVendor:",
        "Customer: Looking for wholesale electronics components.\nVendor:",
        "Customer: Need printing services for marketing materials, 10,000 units.\nVendor:",
    ]

    selected_sample = st.selectbox("Choose a sample prompt", ["Custom..."] + sample_prompts)

    if selected_sample == "Custom...":
        prompt = st.text_area("Enter your prompt:", height=100)
    else:
        prompt = st.text_area("Prompt:", value=selected_sample, height=100)

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1,
                                help="Higher = more creative/random")
    with col2:
        max_tokens = st.slider("Max new tokens", 50, 300, 150)

    st.divider()

    if st.button("Generate Response", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a prompt first.")
        else:
            adapter_exists = os.path.exists(os.path.join(output_dir, "lora_adapter"))

            if not adapter_exists:
                st.warning("⚠️ No trained adapter found. Showing what the output would look like after training.")
                st.info(f"""
                **What will happen after training:**
                
                Prompt: `{prompt}`
                
                The fine-tuned model will respond with vendor-style quotes like:
                "We supply [product] at ₹[X]/unit for bulk orders. 
                [Delivery time]. [Warranty]. [Payment terms]."
                
                The base model without fine-tuning gives generic text.
                The LoRA-adapted model gives structured vendor responses.
                """)
            else:
                with st.spinner("Loading model and generating..."):
                    try:
                        from lora_core import LoRAPipeline
                        pipeline = LoRAPipeline(model_path, output_dir=output_dir)
                        pipeline.load_base_model()
                        pipeline.load_adapter(os.path.join(output_dir, "lora_adapter"))

                        response = pipeline.generate(
                            prompt,
                            max_new_tokens=max_tokens,
                            temperature=temperature
                        )

                        st.success("Fine-tuned model response:")
                        st.write(response)

                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.info("Make sure the model path is correct and the adapter has been trained.")

    with st.expander("📖 How to use this in an interview"):
        st.markdown("""
        **When Madan asks: "What have you built with LLMs?"**
        
        Say:
        > "I built a full LoRA fine-tuning pipeline for domain adaptation. 
        > I took a base model (phi-2), applied LoRA adapters targeting only the attention layers,
        > trained it on vendor-quote style data using PEFT, and evaluated it with ROUGE metrics.
        > The adapter is only ~20MB vs 5.5GB for the full model — so it's deployable anywhere.
        > Here's the Streamlit UI I built to demonstrate it."
        
        **Key terms to drop naturally:**
        - LoRA rank (r=16), alpha, effective scale
        - PEFT library, task vectors
        - ROUGE-L evaluation
        - Adapter vs full fine-tuning memory comparison
        - Causal language modeling objective
        """)
