"""
Generate Q&A training pairs from paper abstracts and parsed PDFs.
Uses Groq API (free Llama 3.3 70B) to create instruction-tuning data.
"""
import json
import os
import time
import re
from groq import Groq

def generate_qa_from_text(client, text, paper_title="", num_pairs=3):
    """Send text to Groq, get back Q&A pairs. Never gives up on rate limits."""
    prompt = f"""You are a scientific expert creating training data for an AI model.
Read this paper text and generate exactly {num_pairs} question-answer pairs.

Rules:
- Questions must be specific and answerable from the text
- Answers must be detailed, factual, and 2-4 sentences long
- Cover different aspects: methods, results, findings, comparisons
- Use proper scientific terminology
- Do NOT mention "the paper" or "the study" in answers - state facts directly

PAPER TITLE: {paper_title}

TEXT:
{text[:3000]}

Respond with ONLY a JSON array, no other text:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""

    backoff = 30  # start at 30 seconds
    max_backoff = 300  # cap at 5 minutes
    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()

            # Clean JSON
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

            pairs = json.loads(content)

            if isinstance(pairs, list) and len(pairs) > 0:
                valid = []
                for p in pairs:
                    if isinstance(p, dict) and "question" in p and "answer" in p:
                        if len(p["question"]) > 10 and len(p["answer"]) > 20:
                            valid.append(p)
                if valid:
                    return valid

            # Got response but bad JSON — retry with shorter backoff
            print(f"    Bad JSON response, retrying...")
            time.sleep(5)

        except json.JSONDecodeError:
            print(f"    JSON parse error, retrying...")
            time.sleep(5)

        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "rate" in error_str or "too many" in error_str:
                print(f"    Rate limited, waiting {backoff}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)  # exponential backoff
            elif "503" in str(e) or "overloaded" in error_str:
                print(f"    Server overloaded, waiting {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            else:
                print(f"    Error: {str(e)[:80]}")
                time.sleep(5)

    return None

def process_abstracts(client, papers_path, output_path):
    """Generate Q&A pairs from paper abstracts."""
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

    for i, paper in enumerate(papers):
        paper_id = paper.get("id", f"paper_{i}")
        if paper_id in processed_ids:
            skipped += 1
            continue

        abstract = paper.get("abstract", "")
        title = paper.get("title", "")

        if not abstract or len(abstract) < 100:
            continue

        pairs = generate_qa_from_text(client, abstract, title, num_pairs=3)

        if pairs:
            for p in pairs:
                existing.append({
                    "instruction": p["question"],
                    "input": "",
                    "output": p["answer"],
                    "source_id": paper_id,
                    "source_title": title,
                    "source_type": "abstract",
                })
            total_new += len(pairs)
        else:
            failed += 1

        # Save every 25 papers (more frequent saves)
        if (i + 1) % 25 == 0:
            with open(output_path, 'w') as f:
                json.dump(existing, f, indent=2)
            print(f"  Progress: {i+1}/{len(papers)} | New: {total_new} | Failed: {failed} | Skipped: {skipped} | Total: {len(existing)}")

        # Safer rate: ~20 requests per minute
        time.sleep(3.5)

    # Final save
    with open(output_path, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"\n  DONE: {total_new} new pairs | Failed: {failed} | Total: {len(existing)}")
    return existing

def process_pdfs(client, parsed_pdfs_path, output_path):
    """Generate Q&A pairs from parsed PDF full text."""
    with open(parsed_pdfs_path) as f:
        papers = json.load(f)

    print(f"  Total parsed PDFs: {len(papers)}")

    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        print(f"  Existing pairs: {len(existing)}")
    else:
        existing = []

    processed_ids = {p.get("source_id", "") for p in existing}
    total_new = 0
    failed = 0

    for i, paper in enumerate(papers):
        paper_id = paper.get("paper_id", f"pdf_{i}")
        if paper_id in processed_ids:
            continue

        text = paper.get("text", "")
        if not text or len(text) < 200:
            continue

        num_pairs = 5 if len(text.split()) > 2000 else 3

        pairs = generate_qa_from_text(client, text, paper_id, num_pairs=num_pairs)

        if pairs:
            for p in pairs:
                existing.append({
                    "instruction": p["question"],
                    "input": "",
                    "output": p["answer"],
                    "source_id": paper_id,
                    "source_title": "",
                    "source_type": "full_text",
                })
            total_new += len(pairs)
        else:
            failed += 1

        if (i + 1) % 25 == 0:
            with open(output_path, 'w') as f:
                json.dump(existing, f, indent=2)
            print(f"  Progress: {i+1}/{len(papers)} | New: {total_new} | Failed: {failed} | Total: {len(existing)}")

        time.sleep(3.5)

    with open(output_path, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"\n  DONE: {total_new} new pairs from PDFs | Total: {len(existing)}")
    return existing

def combine_all(output_dir, final_path):
    """Combine all Q&A pairs into one training file."""
    all_pairs = []

    for filename in sorted(os.listdir(output_dir)):
        if filename.startswith("qa_") and filename.endswith(".json"):
            path = os.path.join(output_dir, filename)
            with open(path) as f:
                pairs = json.load(f)
            all_pairs.extend(pairs)
            print(f"  {filename}: {len(pairs)} pairs")

    seen = set()
    unique = []
    for p in all_pairs:
        q = p["instruction"].lower().strip()
        if q not in seen:
            seen.add(q)
            unique.append(p)

    with open(final_path, 'w') as f:
        json.dump(unique, f, indent=2)

    print(f"\n  Total: {len(all_pairs)} | Unique: {len(unique)}")
    print(f"  Saved to: {final_path}")

def main():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY environment variable")
        return

    client = Groq(api_key=api_key)
    os.makedirs("data/training", exist_ok=True)

    print("=" * 60)
    print("Q&A Training Pair Generator")
    print("=" * 60)

    pubmed_path = "data/raw/pubmed/pubmed_papers.json"
    if os.path.exists(pubmed_path):
        print(f"\n--- PubMed Abstracts ---")
        process_abstracts(client, pubmed_path, "data/training/qa_pubmed.json")

    arxiv_path = "data/raw/arxiv/arxiv_papers.json"
    if os.path.exists(arxiv_path):
        print(f"\n--- ArXiv Abstracts ---")
        process_abstracts(client, arxiv_path, "data/training/qa_arxiv.json")

    parsed_path = "data/training/parsed_arxiv_pdfs.json"
    if os.path.exists(parsed_path):
        print(f"\n--- ArXiv Full Text (PDFs) ---")
        process_pdfs(client, parsed_path, "data/training/qa_pdfs.json")

    print(f"\n--- Combining All ---")
    combine_all("data/training", "data/training/training_data_final.json")

if __name__ == "__main__":
    main()
