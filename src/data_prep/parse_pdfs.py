"""
PDF Parser for Hydro-Finetune
Extracts clean text from scientific PDFs.
Removes references, captions, headers, footers.
"""
import fitz  # pymupdf
import json
import os
import re
import time

def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text.strip())
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        return None

def clean_scientific_text(raw_text):
    """Clean extracted text: remove references, captions, headers."""
    if not raw_text:
        return None
    
    text = raw_text
    
    # Step 1: Remove everything after "References" section
    # Common patterns for reference sections
    ref_patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*Works Cited\s*\n',
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, text)
        if match:
            # Keep everything before references
            text = text[:match.start()]
            break
    
    # Step 2: Remove figure and table captions
    text = re.sub(r'Figure\s+\d+[\.:].+?(?=\n)', '', text)
    text = re.sub(r'Fig\.\s*\d+[\.:].+?(?=\n)', '', text)
    text = re.sub(r'Table\s+\d+[\.:].+?(?=\n)', '', text)
    
    # Step 3: Remove common headers/footers
    text = re.sub(r'https?://\S+', '', text)  # URLs
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)  # DOIs
    text = re.sub(r'©.*?\d{4}.*?\n', '', text)  # Copyright lines
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)  # Dates in headers
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    
    # Step 4: Remove page numbers (standalone numbers on a line)
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    
    # Step 5: Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' {2,}', ' ', text)  # Max 1 space
    text = re.sub(r'\t', ' ', text)  # Tabs to spaces
    
    # Step 6: Remove very short lines (likely headers/footers)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep lines with actual content (more than 20 chars)
        # or empty lines for paragraph breaks
        if len(stripped) > 20 or stripped == '':
            cleaned_lines.append(stripped)
    text = '\n'.join(cleaned_lines)
    
    # Step 7: Final cleanup
    text = text.strip()
    
    # Minimum length check - skip if too short after cleaning
    if len(text) < 200:
        return None
    
    return text

def parse_all_pdfs(pdf_dir, output_path, limit=None):
    """Parse all PDFs in a directory and save clean text."""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if limit:
        pdf_files = pdf_files[:limit]
    
    print(f"Found {len(pdf_files)} PDFs to parse")
    
    parsed = []
    failed = 0
    skipped = 0
    
    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        
        # Extract raw text
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            failed += 1
            continue
        
        # Clean the text
        clean_text = clean_scientific_text(raw_text)
        if not clean_text:
            skipped += 1
            continue
        
        # Get paper ID from filename
        paper_id = pdf_file.replace('.pdf', '')
        
        parsed.append({
            "paper_id": paper_id,
            "filename": pdf_file,
            "text": clean_text,
            "word_count": len(clean_text.split()),
            "char_count": len(clean_text),
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(pdf_files)} | Parsed: {len(parsed)} | Failed: {failed} | Skipped: {skipped}")
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    
    # Print summary
    total_words = sum(p['word_count'] for p in parsed)
    avg_words = total_words / len(parsed) if parsed else 0
    
    print(f"\n{'='*60}")
    print(f"PARSING COMPLETE")
    print(f"  Total PDFs: {len(pdf_files)}")
    print(f"  Successfully parsed: {len(parsed)}")
    print(f"  Failed (corrupt/unreadable): {failed}")
    print(f"  Skipped (too short after cleaning): {skipped}")
    print(f"  Total words extracted: {total_words:,}")
    print(f"  Average words per paper: {avg_words:.0f}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}")
    
    return parsed

def main():
    print("="*60)
    print("PDF Parser for Hydro-Finetune")
    print("="*60)
    
    # Parse ArXiv PDFs
    arxiv_pdf_dir = "data/raw/arxiv/pdfs"
    if os.path.exists(arxiv_pdf_dir):
        print(f"\n--- Parsing ArXiv PDFs ---")
        parse_all_pdfs(
            pdf_dir=arxiv_pdf_dir,
            output_path="data/training/parsed_arxiv_pdfs.json"
        )
    else:
        print(f"No ArXiv PDFs found at {arxiv_pdf_dir}")
    
    # Parse EarthArXiv PDFs
    ea_pdf_dir = "data/raw/eartharxiv/pdfs"
    if os.path.exists(ea_pdf_dir) and os.listdir(ea_pdf_dir):
        print(f"\n--- Parsing EarthArXiv PDFs ---")
        parse_all_pdfs(
            pdf_dir=ea_pdf_dir,
            output_path="data/training/parsed_eartharxiv_pdfs.json"
        )
    else:
        print("No EarthArXiv PDFs found")

if __name__ == "__main__":
    main()
