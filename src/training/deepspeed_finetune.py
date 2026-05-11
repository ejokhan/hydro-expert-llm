"""
DeepSpeed ZeRO-2 Full Fine-tuning of Mistral-7B
Multi-GPU (3x A100) comparison against QLoRA single-GPU baseline.
"""
import json
import os
import time
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def load_data(data_path, val_split=0.05):
    with open(data_path) as f:
        data = json.load(f)
    formatted = []
    for item in data:
        if len(item["instruction"]) >= 10 and len(item["output"]) >= 20:
            formatted.append({"text": f"<s>[INST] {item['instruction']} [/INST] {item['output']}</s>"})
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=val_split, seed=42)
    print(f"Train: {len(split['train'])} | Val: {len(split['test'])}")
    return split["train"], split["test"]


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    data_path = "data/training/qa_pubmed_gpu.json"
    output_dir = "outputs/hydro-mistral-deepspeed"

    print("=" * 60)
    print("DeepSpeed ZeRO-2 Full Fine-tuning")
    print("=" * 60)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        print(f"[Rank {local_rank}] GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"[Rank {local_rank}] Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
        print(f"[Rank {local_rank}] Total GPUs: {torch.cuda.device_count()}")

    print(f"\n[1/3] Loading data...")
    train_data, val_data = load_data(data_path)

    print(f"\n[2/3] Loading model in bf16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {trainable:,} (100% — full fine-tune)")

    print(f"\n[3/3] Starting training with DeepSpeed ZeRO-2...")
    start_time = time.time()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        deepspeed="configs/ds_zero2.json",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
    )

    trainer.train()
    total_time = time.time() - start_time

    eval_results = trainer.evaluate()

    if local_rank == 0:
        trainer.save_model(os.path.join(output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

        print(f"\n{'=' * 60}")
        print(f"DeepSpeed Training Complete!")
        print(f"Training time: {total_time/3600:.1f} hours")
        print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
        print(f"Model saved to: {output_dir}/final_model")
        print(f"{'=' * 60}")

        comparison = {
            "method": "DeepSpeed ZeRO-2 Full Fine-tune",
            "model": model_name,
            "gpus": torch.cuda.device_count(),
            "training_time_hours": total_time / 3600,
            "eval_loss": eval_results["eval_loss"],
            "trainable_params": trainable,
            "trainable_pct": 100.0,
        }
        with open("data/evaluation/deepspeed_results.json", "w") as f:
            json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    main()
