# 🧬 Hydro Expert LLM: Domain-Specific LLM Fine-tuning for Hydrology

Fine-tuned Mistral-7B using QLoRA on 41,958 hydrology Q&A pairs to create a domain-expert language model for water science.

## Key Results

### Training Metrics

| Metric | Start | End |
|--------|-------|-----|
| Loss | 2.176 | 1.232 |
| Token Accuracy | 59.7% | 70.4% |
| Eval Loss | — | 1.502 |
| Training Time | — | 4h 10m on A100 |

### Base vs Fine-tuned Comparison

Tested on 10 hydrology questions without RAG context (pure knowledge test):

| Metric | Base Mistral-7B | Fine-tuned |
|--------|:---:|:---:|
| Avg Response Time | 18.5s | 6.7s |
| Answer Style | Generic, textbook | Concise, domain-expert |
| Specificity | General ML concepts | Specific methods, metrics, results |
| Speed Improvement | — | **3-4x faster** |

**Example — "Compare random forest and gradient boosting for water quality classification":**

**Base:** *"Random Forest and Gradient Boosting are two popular machine learning algorithms used for classification problems... Each decision tree in the forest..."* (generic textbook explanation)

**Fine-tuned:** *"Random forest performed better than gradient boosting... achieved accuracy of 98.8%"* (specific finding, cites numbers)

## Architecture
41,958 Q&A Pairs (generated from 29,654 scientific papers)
|
v
Mistral-7B loaded in 4-bit (NF4 quantization via bitsandbytes)
|
v
LoRA adapters attached (r=16, alpha=32, 1.1% trainable params)
|
v
3 epochs training (bf16, cosine LR schedule, lr=5e-5)
|
v
87.6 MB adapter file (outputs/hydro-mistral-qlora/final_adapter/)
## Data Pipeline

### Paper Collection
- **29,654 abstracts** from PubMed (20,292), ArXiv (8,165), Semantic Scholar (1,182), USGS (15), EarthArXiv (62)
- **8,102 ArXiv PDFs** downloaded and parsed with PyMuPDF → 50 million words

### Q&A Generation
- Used Qwen2.5-3B on TACC A100 GPU to generate training pairs
- Each paper abstract → 3 question-answer pairs
- 41,958 pairs generated at 298 papers/hour, 98.5% success rate
- Format: instruction (question) + output (detailed scientific answer)

### Fine-tuning Configuration
| Parameter | Value |
|-----------|-------|
| Base Model | Mistral-7B-Instruct-v0.3 |
| Method | QLoRA (4-bit NF4 + LoRA) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Targets | q,k,v,o,gate,up,down projections |
| Trainable Params | 41.9M / 3.8B (1.1%) |
| Batch Size | 16 (4 × 4 gradient accumulation) |
| Learning Rate | 5e-5 (cosine schedule) |
| Epochs | 3 |
| Precision | bf16 |
| GPU | NVIDIA A100 40GB |

## Project Structure
hydro-expert-llm/
├── src/
│   ├── data_prep/
│   │   ├── collect_phase1.py       # Paper collection (PubMed, ArXiv, etc.)
│   │   ├── finish_downloads.py     # ArXiv PDF downloader
│   │   ├── parse_pdfs.py           # PDF text extraction
│   │   ├── generate_qa_pairs.py    # Q&A generation (Groq API)
│   │   └── generate_qa_gpu.py      # Q&A generation (local GPU)
│   ├── training/
│   │   └── finetune_qlora.py       # QLoRA fine-tuning script
│   └── evaluation/
│       └── evaluate_finetune.py    # Base vs fine-tuned comparison
├── scripts/
│   ├── run_qa_generation.slurm     # GPU Q&A generation job
│   ├── run_finetune.slurm          # Fine-tuning job
│   └── run_eval_finetune.slurm     # Evaluation job
├── data/
│   ├── raw/                        # Collected papers
│   ├── training/                   # Q&A pairs + parsed PDFs
│   └── evaluation/                 # Base vs fine-tuned results
└── outputs/
└── hydro-mistral-qlora/        # LoRA adapter weights
## Quick Start

```bash
git clone https://github.com/ejokhan/hydro-expert-llm.git
cd hydro-expert-llm
pip install torch transformers peft trl bitsandbytes datasets

# Generate Q&A pairs (requires GPU)
python src/data_prep/generate_qa_gpu.py

# Fine-tune (requires GPU)
python src/training/finetune_qlora.py

# Evaluate
python src/evaluation/evaluate_finetune.py
```

## Connection to HydroRAG

This is **Project 2** in a 3-project LLM portfolio:

1. **[HydroRAG](https://github.com/ejokhan/hydro-rag)** — RAG system over 8,618 papers with 15-config benchmark
2. **Hydro Expert LLM** (this repo) — Domain-specific LLM fine-tuning
3. **HydroAgent** (planned) — Agentic AI for autonomous hydrological analysis

The fine-tuned model can replace the base LLM in HydroRAG for higher quality domain-specific answers.

## Tech Stack

Python, PyTorch, Hugging Face Transformers, PEFT, TRL, bitsandbytes, NVIDIA A100 GPU on TACC Lonestar6 via NSF NAIRR Pilot.

## Author

**Ijaz Ul Haq, Ph.D.** — AI/ML Research Scientist

University of Vermont | Water Resources Institute

[Google Scholar](https://scholar.google.com/citations?user=qHTMlKIAAAAJ&hl=en) | [GitHub](https://github.com/ejokhan)

## License

MIT
