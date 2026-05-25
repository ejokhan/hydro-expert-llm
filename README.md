# 🧬 Hydro Expert LLM: Domain-Specific LLM Fine-tuning for Hydrology

Fine-tuned Mistral-7B using QLoRA on 41,958 hydrology Q&A pairs to create a domain-expert language model for water science. Includes distributed hyperparameter search with Ray Tune (24 configurations) and DeepSpeed ZeRO-2 multi-GPU comparison.

**[HuggingFace Model](https://huggingface.co/Ejokhan/hydro-expert-llm)** | **[HydroRAG (Project 1)](https://github.com/ejokhan/hydro-rag)** | **[Live RAG Demo](https://hydrorag.streamlit.app)**

## Key Results

### QLoRA vs DeepSpeed Full Fine-tuning

| Method | GPUs | Params Trained | Eval Loss | Time |
|--------|:---:|:---:|:---:|:---:|
| QLoRA r=16 (hand-tuned) | 1 x A100 | 41M (1.1%) | 1.502 | 4h |
| **QLoRA r=64 (Ray Tune best)** | **1 x A100** | **~160M (4.2%)** | **1.469** | **~4h** |
| DeepSpeed ZeRO-2 Full FT | 3 x A100 | 7.2B (100%) | 1.492 | 3.5h |

**Finding:** QLoRA r=64 on a single GPU outperformed DeepSpeed ZeRO-2 full fine-tuning on 3 GPUs while training 45x fewer parameters.

### Ray Tune Hyperparameter Sweep (24 configurations)

Distributed search across 3 parallel A100 GPUs varying LoRA rank, learning rate, and dropout:

| Rank | LoRA r | Learning Rate | Dropout | Eval Loss |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 64 | 5e-5 | 0.0 | **1.469** |
| 2 | 64 | 5e-5 | 0.05 | 1.470 |
| 3 | 32 | 1e-4 | 0.0 | 1.472 |
| 4 | 32 | 1e-4 | 0.05 | 1.473 |
| ... | ... | ... | ... | ... |
| 23 | 8 | 1e-5 | 0.0 | 1.538 |
| 24 | 8 | 1e-5 | 0.05 | 1.539 |

**Findings:**
- Higher LoRA rank consistently improves performance: r=64 > r=32 > r=16 > r=8
- Dropout has negligible effect (0.0 vs 0.05 = <0.1% difference)
- Optimal learning rate is 5e-5 for high ranks, 1e-4 for lower ranks
- Ray Tune best (r=64) improved eval loss by 2.2% over hand-tuned baseline (r=16)

### Base vs Fine-tuned Comparison

Tested on 10 hydrology questions without RAG context (pure knowledge test):

| Metric | Base Mistral-7B | Fine-tuned |
|--------|:---:|:---:|
| Avg Response Time | 18.5s | 6.7s |
| Answer Style | Generic, textbook | Concise, domain-expert |
| Loss | 2.176 | 1.232 (down 43%) |
| Token Accuracy | 59.7% | 70.4% |
| Speed Improvement | - | **2.8x faster** |

**Example - "Compare random forest and gradient boosting for water quality classification":**

**Base:** *"Random Forest and Gradient Boosting are two popular machine learning algorithms used for classification problems... Each decision tree in the forest..."* (generic textbook)

**Fine-tuned:** *"Random forest performed better than gradient boosting... achieved accuracy of 98.8%"* (specific finding, cites metrics)

## Architecture
41,958 Q&A Pairs (generated from 29,654 scientific papers)
|
v
Mistral-7B loaded in 4-bit (NF4 quantization via bitsandbytes)
|
v
LoRA adapters attached (r=16/32/64, alpha=2r, 1.1-4.2% trainable)
|
v
Training (bf16, cosine LR, paged AdamW 32-bit)
|
v
87.6 MB adapter file -> published on HuggingFace Hub

## Data Pipeline

### Paper Collection
- **29,654 abstracts** from PubMed (20,292), ArXiv (8,165), Semantic Scholar (1,182), USGS (15), EarthArXiv (62)
- **8,102 ArXiv PDFs** downloaded and parsed with PyMuPDF -> 50 million words

### Q&A Generation
- Used Qwen2.5-3B on TACC A100 GPU (no API rate limits)
- Each paper abstract -> 3 question-answer pairs
- 41,958 pairs at 298 papers/hour, 98.5% success rate

### Fine-tuning Configurations

| Parameter | Hand-tuned | Ray Tune Best |
|-----------|:---:|:---:|
| Base Model | Mistral-7B-Instruct-v0.3 | Same |
| Method | QLoRA (4-bit NF4 + LoRA) | Same |
| LoRA Rank | 16 | **64** |
| LoRA Alpha | 32 | **128** |
| LoRA Dropout | 0.05 | **0.0** |
| Learning Rate | 5e-5 | 5e-5 |
| Batch Size | 16 | 16 |
| Epochs | 3 | 1 (sweep) |
| Precision | bf16 | bf16 |
| Eval Loss | 1.502 | **1.469** |

### DeepSpeed ZeRO-2 Configuration

| Parameter | Value |
|-----------|-------|
| ZeRO Stage | 2 |
| Optimizer Offload | CPU |
| GPUs | 3 x A100 40GB |
| Gradient Checkpointing | Enabled |
| Trainable Params | 7.2B (100%) |
| Eval Loss | 1.492 |

## Project Structure

```
hydro-expert-llm/
├── src/
│   ├── data_prep/
│   │   ├── collect_phase1.py         # Paper collection
│   │   ├── finish_downloads.py       # ArXiv PDF downloader
│   │   ├── parse_pdfs.py             # PDF text extraction
│   │   ├── generate_qa_pairs.py      # Q&A generation (Groq API)
│   │   └── generate_qa_gpu.py        # Q&A generation (local GPU)
│   ├── training/
│   │   ├── finetune_qlora.py         # QLoRA fine-tuning
│   │   ├── ray_tune_qlora.py         # Ray Tune hyperparameter sweep
│   │   └── deepspeed_finetune.py     # DeepSpeed ZeRO-2 full fine-tuning
│   └── evaluation/
│       └── evaluate_finetune.py      # Base vs fine-tuned comparison
├── configs/
│   └── ds_zero2.json                 # DeepSpeed ZeRO-2 config
├── scripts/
│   ├── run_qa_generation.slurm       # GPU Q&A generation job
│   ├── run_finetune.slurm            # QLoRA fine-tuning job
│   ├── run_ray_tune.slurm            # Ray Tune sweep job
│   ├── run_deepspeed.slurm           # DeepSpeed multi-GPU job
│   └── run_eval_finetune.slurm       # Evaluation job
├── data/
│   └── evaluation/
│       ├── base_vs_finetuned.json    # 10-question comparison
│       └── ray_tune_results.json     # 24-config sweep results
└── outputs/
└── hydro-mistral-qlora/          # LoRA adapter weights
```

## Quick Start

```bash
git clone https://github.com/ejokhan/hydro-expert-llm.git
cd hydro-expert-llm
pip install torch transformers peft trl bitsandbytes datasets ray deepspeed

# Generate Q&A pairs (requires GPU)
python src/data_prep/generate_qa_gpu.py

# Fine-tune with QLoRA (single GPU)
python src/training/finetune_qlora.py

# Ray Tune hyperparameter sweep (multi-GPU)
python src/training/ray_tune_qlora.py

# DeepSpeed full fine-tune (multi-GPU)
torchrun --nproc_per_node=3 src/training/deepspeed_finetune.py

# Evaluate base vs fine-tuned
python src/evaluation/evaluate_finetune.py
```

## Use the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config, device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "Ejokhan/hydro-expert-llm")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

question = "What ML models are most effective for streamflow prediction?"
messages = [{"role": "user", "content": question}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.3, do_sample=True)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Connection to HydroRAG

This is **Project 2** in a 3-project LLM portfolio:

1. **[HydroRAG](https://github.com/ejokhan/hydro-rag)** - RAG system with 15-config benchmark over 8,618 papers. [Live demo](https://hydrorag.streamlit.app)
2. **Hydro Expert LLM** (this repo) - Domain-specific fine-tuning with distributed hyperparameter search
3. **HydroAgent** (planned) - Agentic AI for autonomous hydrological analysis

## Tech Stack

Python, PyTorch, Hugging Face (Transformers, PEFT, TRL, Hub), bitsandbytes, Ray Tune, DeepSpeed, NVIDIA A100 GPUs on TACC Lonestar6 via NSF NAIRR Pilot.

## Author

**Ijaz Ul Haq, Ph.D.** - AI/ML Research Scientist

University of Vermont | Water Resources Institute

[Google Scholar](https://scholar.google.com/citations?user=qHTMlKIAAAAJ&hl=en) | [GitHub](https://github.com/ejokhan)

## License

MIT
