"""
Ray Tune Hyperparameter Sweep for QLoRA Fine-tuning
Runs parallel trials on 3 A100 GPUs varying LoRA rank, learning rate, dropout.
"""
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


def load_data(data_path):
    with open(data_path) as f:
        data = json.load(f)
    formatted = []
    for item in data:
        if len(item["instruction"]) >= 10 and len(item["output"]) >= 20:
            formatted.append({"text": f"<s>[INST] {item['instruction']} [/INST] {item['output']}</s>"})
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]


def train_with_config(config):
    """Single training run with given hyperparameters."""
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    data_path = "/work/10655/ejokhan123/ls6/hydro-finetune/data/training/qa_pubmed_gpu.json"
    output_dir = f"/work/10655/ejokhan123/ls6/hydro-finetune/outputs/ray_trial_{config['lora_r']}_{config['lr']:.0e}"

    train_data, val_data = load_data(data_path)

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
        model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_r"] * 2,
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Config: r={config['lora_r']}, lr={config['lr']}, dropout={config['lora_dropout']}")
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="no",
        bf16=True,
        report_to="none",
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # Report to Ray
    tune.report(
        eval_loss=eval_results["eval_loss"],
        train_loss=trainer.state.log_history[-2].get("loss", 999),
        lora_r=config["lora_r"],
        lr=config["lr"],
        dropout=config["lora_dropout"],
    )


def main():
    print("=" * 60)
    print("Ray Tune — QLoRA Hyperparameter Sweep")
    print("=" * 60)

    os.environ["RAY_TMPDIR"] = "/work/10655/ejokhan123/ls6/hydro-finetune/ray_tmp"
    os.makedirs(os.environ["RAY_TMPDIR"], exist_ok=True)

    ray.init(num_gpus=3)

    search_space = {
        "lora_r": tune.grid_search([8, 16, 32, 64]),
        "lr": tune.grid_search([1e-5, 5e-5, 1e-4]),
        "lora_dropout": tune.grid_search([0.0, 0.05]),
    }

    scheduler = ASHAScheduler(
        metric="eval_loss",
        mode="min",
        max_t=1,
        grace_period=1,
    )

    results = tune.run(
        train_with_config,
        config=search_space,
        resources_per_trial={"gpu": 1},
        num_samples=1,
        scheduler=scheduler,
        storage_path="/work/10655/ejokhan123/ls6/hydro-finetune/ray_results",
        verbose=1,
    )

    # Print results
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("=" * 60)

    best = results.get_best_result("eval_loss", "min")
    print(f"\nBest config:")
    print(f"  LoRA rank: {best.config['lora_r']}")
    print(f"  Learning rate: {best.config['lr']}")
    print(f"  Dropout: {best.config['lora_dropout']}")
    print(f"  Eval loss: {best.metrics['eval_loss']:.4f}")
    print(f"  Train loss: {best.metrics['train_loss']:.4f}")

    # Save summary
    summary = []
    for r in results:
        summary.append({
            "lora_r": r.config["lora_r"],
            "lr": r.config["lr"],
            "dropout": r.config["lora_dropout"],
            "eval_loss": r.metrics.get("eval_loss", None),
            "train_loss": r.metrics.get("train_loss", None),
        })

    summary.sort(key=lambda x: x.get("eval_loss", 999))

    with open("data/evaluation/ray_tune_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll results (sorted by eval_loss):")
    for s in summary:
        print(f"  r={s['lora_r']:>2}, lr={s['lr']:.0e}, drop={s['dropout']}, eval_loss={s.get('eval_loss', 'N/A')}")

    print(f"\nResults saved to data/evaluation/ray_tune_results.json")
    ray.shutdown()


if __name__ == "__main__":
    main()
