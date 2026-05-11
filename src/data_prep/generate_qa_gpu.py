"""
Generate Q&A training pairs using local GPU model.
No API, no rate limits. Uses Qwen2.5-3B on A100.
"""
import json
import os
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded!")
    return model, tokenizer

def generate_qa(model, tokenizer, text, title="", num_pairs=3):
    prompt = f"""You are a scientific expert. Read this paper and generate exactly {num_pairs} question-answer pairs.

Rules:
- Questions must be specific and answerable from the text
- Answers must be detailed, 2-4 sentences
- Use scientific terminology
- Do NOT mention "the paper" or "the study"

TITLE: {title}
TEXT: {text[:2000]}

Respond with ONLY a JSON array:
[{{"question": "...", "answer": "..."}}]"""

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Clean and parse JSON
    response = re.sub(r'^```json\s*', '', response)
    response = re.sub(r'\s*```$', '', response)
    
    try:
        pairs = json.loads(response)
        if isinstance(pairs, list):
            valid = []
            for p in pairs:
                if isinstance(p, dict) and "question" in p and "answer" in p:
                    if len(p["question"]) > 10 and len(p["answer"]) > 20:
                        valid.append(p)
            if valid:
                return valid
    except json.JSONDecodeError:
        pass
    
    return None

def process_papers(model, tokenizer, papers_path, output_path, source_type="abstract"):
    with open(papers_path) as f:
        papers = json.load(f)
    
    print(f"  Total papers: {len(papers)}")
    
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        print(f"  Existing pairs: {len(existing)}")
    else:
        existing = []
    
    processed_ids = {p.get("source_id", "") for p in existing}
    total_new = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    
    for i, paper in enumerate(papers):
        paper_id = paper.get("id", paper.get("paper_id", f"paper_{i}"))
        if paper_id in processed_ids:
            skipped += 1
            continue
        
        text = paper.get("abstract", paper.get("text", ""))
        title = paper.get("title", "")
        
        if not text or len(text) < 100:
            continue
        
        pairs = generate_qa(model, tokenizer, text, title, num_pairs=3)
        
        if pairs:
            for p in pairs:
                existing.append({
                    "instruction": p["question"],
                    "input": "",
                    "output": p["answer"],
                    "source_id": paper_id,
                    "source_title": title,
                    "source_type": source_type,
                })
            total_new += len(pairs)
        else:
            failed += 1
        
        # Save every 100 papers
        if (i + 1) % 100 == 0:
            with open(output_path, 'w') as f:
                json.dump(existing, f, indent=2)
            elapsed = time.time() - start_time
            rate = (i + 1 - skipped) / elapsed * 3600
            print(f"  [{i+1}/{len(papers)}] New: {total_new} | Failed: {failed} | Skipped: {skipped} | Rate: {rate:.0f}/hr | Total: {len(existing)}")
    
    with open(output_path, 'w') as f:
        json.dump(existing, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n  DONE in {elapsed/60:.1f}min: {total_new} new pairs | Failed: {failed} | Total: {len(existing)}")

def main():
    model, tokenizer = load_model()
    os.makedirs("data/training", exist_ok=True)
    
    print("=" * 60)
    print("Q&A Generation (LOCAL GPU - No API limits)")
    print("=" * 60)
    
    pubmed_path = "data/raw/pubmed/pubmed_papers.json"
    if os.path.exists(pubmed_path):
        print(f"\n--- PubMed Abstracts ---")
        process_papers(model, tokenizer, pubmed_path, "data/training/qa_pubmed_gpu.json", "abstract")
    
    arxiv_path = "data/raw/arxiv/arxiv_papers.json"
    if os.path.exists(arxiv_path):
        print(f"\n--- ArXiv Abstracts ---")
        process_papers(model, tokenizer, arxiv_path, "data/training/qa_arxiv_gpu.json", "abstract")
    
    print("\n--- Combining All ---")
    all_pairs = []
    for f_name in sorted(os.listdir("data/training")):
        if f_name.startswith("qa_") and f_name.endswith(".json"):
            with open(os.path.join("data/training", f_name)) as f:
                pairs = json.load(f)
            all_pairs.extend(pairs)
            print(f"  {f_name}: {len(pairs)} pairs")
    
    seen = set()
    unique = [p for p in all_pairs if p["instruction"].lower().strip() not in seen and not seen.add(p["instruction"].lower().strip())]
    
    with open("data/training/training_data_final.json", 'w') as f:
        json.dump(unique, f, indent=2)
    print(f"\n  Total: {len(all_pairs)} | Unique: {len(unique)}")

if __name__ == "__main__":
    main()
