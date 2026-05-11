"""
QLoRA Fine-tuning for Hydrology Domain
Fine-tunes Mistral-7B on 41,958 hydrology Q&A pairs using 4-bit quantization + LoRA adapters.
"""
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_training_data(data_path, val_split=0.05):
    """Load Q&A pairs and split into train/val."""
    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} Q&A pairs")

    formatted = []
    for item in data:
        question = item["instruction"]
        answer = item["output"]
        if len(question) < 10 or len(answer) < 20:
            continue
        text = f"""<s>[INST] {question} [/INST] {answer}</s>"""
        formatted.append({"text": text})

    print(f"After filtering: {len(formatted)} pairs")

    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=val_split, seed=42)
    print(f"Train: {len(split['train'])} | Val: {len(split['test'])}")
    return split["train"], split["test"]


def setup_model(model_name):
    """Load model in 4-bit quantization."""
    print(f"\nLoading {model_name} in 4-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded! GPU memory used: {mem:.1f} GB")

    return model, tokenizer


def setup_lora(model):
    """Attach LoRA adapters to the model."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / total * 100
    print(f"\nTrainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    return model


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    data_path = "data/training/qa_pubmed_gpu.json"
    output_dir = "outputs/hydro-mistral-qlora"
    max_seq_length = 512
    num_epochs = 3
    batch_size = 4
    gradient_accumulation = 4
    learning_rate = 5e-5
    warmup_ratio = 0.03

    print("=" * 60)
    print("HydroFinetune - QLoRA Fine-tuning")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size} x {gradient_accumulation} = {batch_size * gradient_accumulation}")
    print(f"Max seq length: {max_seq_length}")
    print(f"Learning rate: {learning_rate}")

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWARNING: No GPU detected!")
        return

    print("\n[1/4] Loading training data...")
    train_dataset, val_dataset = load_training_data(data_path)

    print("\n[2/4] Loading model...")
    model, tokenizer = setup_model(model_name)

    print("\n[3/4] Setting up LoRA adapters...")
    model = setup_lora(model)

    print("\n[4/4] Starting training...")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    print("\nTraining started!")
    trainer.train()

    final_path = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Adapter saved to: {final_path}")
    adapter_size = sum(
        os.path.getsize(os.path.join(final_path, f))
        for f in os.listdir(final_path)
    ) / 1e6
    print(f"Adapter size: {adapter_size:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
