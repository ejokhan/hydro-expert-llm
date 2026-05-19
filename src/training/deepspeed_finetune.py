"""
DeepSpeed ZeRO-2 Full Fine-tuning of Mistral-7B.

Multi-GPU (3x A100 PCIe 40GB) comparison against the QLoRA single-GPU baseline.

Key changes vs. previous version:
1. Replaced deprecated `torch_dtype=` with `dtype=` in from_pretrained.
2. Added HYDRO_SMOKE=1 env var: subsamples data and caps training at 20 steps
   so we can verify launcher + NCCL + DeepSpeed init in ~3 min instead of
   waiting for a 4+ hour run to discover the same crash.
3. data_path made absolute so the smoke slurm doesn't have to cd.
"""
import json
import os
import time
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


DATA_PATH = "/work/10655/ejokhan123/ls6/hydro-finetune/data/training/qa_pubmed_gpu.json"
OUTPUT_DIR = "/work/10655/ejokhan123/ls6/hydro-finetune/outputs/hydro-mistral-deepspeed"
DS_CONFIG = "/work/10655/ejokhan123/ls6/hydro-finetune/configs/ds_zero3.json"
RESULTS_JSON = "/work/10655/ejokhan123/ls6/hydro-finetune/data/evaluation/deepspeed_results.json"

SMOKE_TEST = os.environ.get("HYDRO_SMOKE", "0") == "1"


def load_data(data_path, val_split=0.05):
    with open(data_path) as f:
        data = json.load(f)
    formatted = []
    for item in data:
        if len(item["instruction"]) >= 10 and len(item["output"]) >= 20:
            formatted.append(
                {"text": f"<s>[INST] {item['instruction']} [/INST] {item['output']}</s>"}
            )
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=val_split, seed=42)
    print(f"Train: {len(split['train'])} | Val: {len(split['test'])}")
    return split["train"], split["test"]


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    print("=" * 60)
    print(f"DeepSpeed ZeRO-2 Full Fine-tuning  (smoke={SMOKE_TEST})")
    print("=" * 60)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        print(f"[Rank {local_rank}] GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"[Rank {local_rank}] Memory: "
              f"{torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
        print(f"[Rank {local_rank}] Total GPUs: {torch.cuda.device_count()}")

    print(f"\n[1/3] Loading data...")
    train_data, val_data = load_data(DATA_PATH)

    if SMOKE_TEST:
        train_data = train_data.select(range(min(200, len(train_data))))
        val_data = val_data.select(range(min(40, len(val_data))))
        print(f"[SMOKE] Subsampled: train={len(train_data)}, val={len(val_data)}")

    print(f"\n[2/3] Loading model in bf16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,           # was torch_dtype= (deprecated)
        trust_remote_code=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {trainable:,} (100% — full fine-tune)")

    print(f"\n[3/3] Starting training with DeepSpeed ZeRO-2...")
    start_time = time.time()

    ta_kwargs = dict(
        output_dir=OUTPUT_DIR,
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
        deepspeed=DS_CONFIG,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
    )
    if SMOKE_TEST:
        ta_kwargs.update(
            max_steps=20,
            warmup_steps=2,
            logging_steps=5,
            eval_steps=20,
            save_strategy="no",
        )

    training_args = TrainingArguments(**ta_kwargs)

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
        if not SMOKE_TEST:
            trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
            tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

        print(f"\n{'=' * 60}")
        print(f"DeepSpeed Training Complete!  (smoke={SMOKE_TEST})")
        print(f"Training time: {total_time/3600:.2f} hours ({total_time:.0f} s)")
        print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
        if not SMOKE_TEST:
            print(f"Model saved to: {OUTPUT_DIR}/final_model")
        print(f"{'=' * 60}")

        comparison = {
            "method": "DeepSpeed ZeRO-2 Full Fine-tune",
            "model": model_name,
            "gpus": torch.cuda.device_count(),
            "training_time_hours": total_time / 3600,
            "eval_loss": float(eval_results["eval_loss"]),
            "trainable_params": trainable,
            "trainable_pct": 100.0,
            "smoke_test": SMOKE_TEST,
        }
        os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
        with open(RESULTS_JSON, "w") as f:
            json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    main()
