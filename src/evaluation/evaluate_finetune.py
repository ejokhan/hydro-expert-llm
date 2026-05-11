"""
Evaluate base vs fine-tuned Mistral-7B on hydrology questions.
Shows side-by-side comparison of answers.
"""
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Hydrology-specific test questions (no RAG context — pure knowledge test)
TEST_QUESTIONS = [
    "What machine learning models are most effective for streamflow prediction?",
    "How is anomaly detection applied to water quality monitoring?",
    "What are the main challenges in deep learning for flood forecasting?",
    "Explain the Nash-Sutcliffe Efficiency metric and when it is used.",
    "How do LSTM networks capture temporal dependencies in hydrological time series?",
    "What is baseflow separation and why is it important in hydrology?",
    "How are transformers being used in rainfall-runoff modeling?",
    "What role does antecedent soil moisture play in flood prediction?",
    "Compare random forest and gradient boosting for water quality classification.",
    "How can transfer learning help with streamflow prediction in ungauged basins?",
]

def load_base_model(model_name):
    """Load base model in 4-bit."""
    print(f"Loading base model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Base model loaded!")
    return model, tokenizer

def load_finetuned_model(base_model, adapter_path):
    """Load fine-tuned adapter on top of base model."""
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("Fine-tuned model loaded!")
    return model

def generate_answer(model, tokenizer, question, max_tokens=300):
    """Generate an answer from the model."""
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response, elapsed

def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    adapter_path = "outputs/hydro-mistral-qlora/final_adapter"

    print("=" * 70)
    print("HydroFinetune — Base vs Fine-tuned Evaluation")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load base model
    print("\n[1/3] Loading base model...")
    base_model, tokenizer = load_base_model(model_name)

    # Generate base answers
    print("\n[2/3] Generating BASE model answers...")
    base_answers = []
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\n  Question {i+1}/{len(TEST_QUESTIONS)}: {q[:60]}...")
        answer, elapsed = generate_answer(base_model, tokenizer, q)
        base_answers.append({"question": q, "answer": answer, "time": elapsed})
        print(f"  Answer ({elapsed:.1f}s): {answer[:150]}...")

    # Load fine-tuned adapter
    print("\n[3/3] Loading fine-tuned adapter and generating answers...")
    ft_model = load_finetuned_model(base_model, adapter_path)

    ft_answers = []
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\n  Question {i+1}/{len(TEST_QUESTIONS)}: {q[:60]}...")
        answer, elapsed = generate_answer(ft_model, tokenizer, q)
        ft_answers.append({"question": q, "answer": answer, "time": elapsed})
        print(f"  Answer ({elapsed:.1f}s): {answer[:150]}...")

    # Print side-by-side comparison
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\n{'='*70}")
        print(f"Q{i+1}: {q}")
        print(f"{'='*70}")
        print(f"\nBASE MODEL ({base_answers[i]['time']:.1f}s):")
        print(f"{base_answers[i]['answer'][:500]}")
        print(f"\nFINE-TUNED ({ft_answers[i]['time']:.1f}s):")
        print(f"{ft_answers[i]['answer'][:500]}")

    # Save results
    os.makedirs("data/evaluation", exist_ok=True)
    results = {
        "base_model": model_name,
        "adapter": adapter_path,
        "questions": []
    }
    for i, q in enumerate(TEST_QUESTIONS):
        results["questions"].append({
            "question": q,
            "base_answer": base_answers[i]["answer"],
            "base_time": base_answers[i]["time"],
            "finetuned_answer": ft_answers[i]["answer"],
            "finetuned_time": ft_answers[i]["time"],
        })

    with open("data/evaluation/base_vs_finetuned.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to data/evaluation/base_vs_finetuned.json")

if __name__ == "__main__":
    main()
