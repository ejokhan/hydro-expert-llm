"""
Ray Tune Hyperparameter Sweep for QLoRA Fine-tuning.

Runs parallel trials on 3 A100 GPUs varying LoRA rank, learning rate, dropout.

Key changes vs. previous version:
1. `ray.train.report({...})` is no longer valid inside a function passed to
   tune.run() in Ray 2.55.1 (it raises DeprecationWarning, which inherits from
   Exception and kills the trial after eval completes). Replaced with a plain
   `return {...}` per the official Ray 2.55.1 Tune docs.
2. Each trial now dumps its metrics to disk *before* returning, so that the
   final aggregation does NOT depend on Ray's result object API (which differs
   between tune.run and tune.Tuner). Even if Ray's downstream API surprises us,
   the data is on disk and recoverable.
3. log_history lookup made robust — was `log_history[-2].get("loss", 999)`,
   which is brittle (it assumes the last entry is eval and second-to-last is a
   train step). Now we scan backwards for the last entry actually containing
   "loss".
4. HYDRO_SMOKE=1 env var enables a 1-trial, 20-step smoke test for ~2 min
   end-to-end verification before committing to the full ~11-hour sweep.
"""
import glob
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
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


TRIAL_DUMP_DIR = "/work/10655/ejokhan123/ls6/hydro-finetune/data/evaluation/ray_trials"
RESULTS_JSON = "/work/10655/ejokhan123/ls6/hydro-finetune/data/evaluation/ray_tune_results.json"
RAY_STORAGE = "/work/10655/ejokhan123/ls6/hydro-finetune/ray_results"
DATA_PATH = "/work/10655/ejokhan123/ls6/hydro-finetune/data/training/qa_pubmed_gpu.json"

SMOKE_TEST = os.environ.get("HYDRO_SMOKE", "0") == "1"


def load_data(data_path):
    with open(data_path) as f:
        data = json.load(f)
    formatted = []
    for item in data:
        if len(item["instruction"]) >= 10 and len(item["output"]) >= 20:
            formatted.append(
                {"text": f"<s>[INST] {item['instruction']} [/INST] {item['output']}</s>"}
            )
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]


def _last_train_loss(log_history):
    """Return the most recent 'loss' from trainer.state.log_history.
    Eval entries have 'eval_loss' but not 'loss'; train-step entries have 'loss'."""
    for entry in reversed(log_history):
        if "loss" in entry:
            return entry["loss"]
    return None


def train_with_config(config):
    """Single training run with given hyperparameters."""
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    output_dir = (
        f"/work/10655/ejokhan123/ls6/hydro-finetune/outputs/"
        f"ray_trial_r{config['lora_r']}_lr{config['lr']:.0e}_d{config['lora_dropout']}"
    )

    train_data, val_data = load_data(DATA_PATH)

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

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_r"] * 2,
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Config: r={config['lora_r']}, lr={config['lr']}, "
        f"dropout={config['lora_dropout']}"
    )
    print(
        f"Trainable: {n_trainable:,} / {n_total:,} "
        f"({n_trainable / n_total * 100:.2f}%)"
    )

    ta_kwargs = dict(
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
    if SMOKE_TEST:
        ta_kwargs.update(
            max_steps=20,
            eval_steps=20,
            warmup_steps=2,
            logging_steps=5,
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
    eval_results = trainer.evaluate()

    metrics = {
        "eval_loss": float(eval_results["eval_loss"]),
        "train_loss": _last_train_loss(trainer.state.log_history),
        "lora_r": config["lora_r"],
        "lr": config["lr"],
        "dropout": config["lora_dropout"],
    }

    os.makedirs(TRIAL_DUMP_DIR, exist_ok=True)
    fname = (
        f"trial_r{config['lora_r']}_lr{config['lr']:.0e}_"
        f"d{config['lora_dropout']}.json"
    )
    with open(os.path.join(TRIAL_DUMP_DIR, fname), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def aggregate_from_disk():
    """Read per-trial JSONs from TRIAL_DUMP_DIR and sort by eval_loss."""
    files = sorted(glob.glob(os.path.join(TRIAL_DUMP_DIR, "trial_*.json")))
    rows = []
    for fp in files:
        with open(fp) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: r.get("eval_loss", float("inf")))
    return rows


def main():
    print("=" * 60)
    print(f"Ray Tune — QLoRA Hyperparameter Sweep  (smoke={SMOKE_TEST})")
    print("=" * 60)

    os.environ["RAY_TMPDIR"] = "/tmp/ray_hydro"
    os.makedirs(os.environ["RAY_TMPDIR"], exist_ok=True)
    os.makedirs(TRIAL_DUMP_DIR, exist_ok=True)

    ray.init(num_gpus=3)

    if SMOKE_TEST:
        search_space = {
            "lora_r": tune.grid_search([8]),
            "lr": tune.grid_search([5e-5]),
            "lora_dropout": tune.grid_search([0.0]),
        }
    else:
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

    tune.run(
        train_with_config,
        config=search_space,
        resources_per_trial={"gpu": 1},
        num_samples=1,
        scheduler=scheduler,
        storage_path=RAY_STORAGE,
        verbose=1,
    )

    summary = aggregate_from_disk()

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("=" * 60)

    if not summary:
        print("No trial JSONs found in", TRIAL_DUMP_DIR)
        print("Something went wrong inside the trainable. Check Ray error files.")
        ray.shutdown()
        return

    best = summary[0]
    print("\nBest config:")
    print(f"  LoRA rank:     {best['lora_r']}")
    print(f"  Learning rate: {best['lr']}")
    print(f"  Dropout:       {best['dropout']}")
    print(f"  Eval loss:     {best['eval_loss']:.4f}")
    print(f"  Train loss:    {best['train_loss']}")

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll results (sorted by eval_loss):")
    for s in summary:
        el = s.get("eval_loss")
        tl = s.get("train_loss")
        el_s = f"{el:.4f}" if el is not None else "N/A"
        tl_s = f"{tl:.4f}" if isinstance(tl, (int, float)) else str(tl)
        print(
            f"  r={s['lora_r']:>2}, lr={s['lr']:.0e}, drop={s['dropout']}, "
            f"eval_loss={el_s}, train_loss={tl_s}"
        )

    print(f"\nResults saved to {RESULTS_JSON}")
    ray.shutdown()


if __name__ == "__main__":
    main()
