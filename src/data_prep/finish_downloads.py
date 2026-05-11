"""
Finish downloading: remaining ArXiv PDFs, USGS PDFs, EarthArXiv
"""
import json
import os
import time
import requests
from datetime import datetime

def download_pdf(url, save_path, timeout=30):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

def download_remaining_arxiv():
    """Download PDFs for ArXiv papers that don't have them yet."""
    with open("data/raw/arxiv/arxiv_papers.json") as f:
        papers = json.load(f)
    
    pdf_dir = "data/raw/arxiv/pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    existing = set(os.listdir(pdf_dir))
    
    need = 0
    downloaded = 0
    failed = 0
    
    for p in papers:
        pdf_name = f"{p['arxiv_id'].replace('/', '_')}.pdf"
        if pdf_name in existing:
            continue
        need += 1
        pdf_url = p.get("pdf_url", "")
        if not pdf_url:
            failed += 1
            continue
        if download_pdf(pdf_url, os.path.join(pdf_dir, pdf_name)):
            downloaded += 1
            existing.add(pdf_name)
        else:
            failed += 1
        if (downloaded + failed) % 50 == 0:
            print(f"  ArXiv PDFs: {downloaded} downloaded, {failed} failed, {need - downloaded - failed} remaining")
        time.sleep(1)
    
    print(f"\n[ArXiv PDFs] Done: {downloaded} new, {failed} failed, {len(existing)} total")

def collect_eartharxiv():
    """Collect EarthArXiv papers."""
    output_dir = "data/raw/eartharxiv"
    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    queries = [
        "streamflow prediction", "flood forecasting",
        "water quality monitoring", "hydrological modeling",
        "groundwater prediction", "drought prediction",
        "anomaly detection water", "sensor data quality",
        "rainfall runoff", "evapotranspiration",
        "snow melt hydrology", "sediment transport",
        "remote sensing water", "climate change water resources",
        "dam safety", "reservoir management",
        "watershed modeling", "baseflow separation",
        "water table prediction", "soil moisture",
    ]
    
    all_papers = []
    seen_ids = set()
    pdfs_downloaded = 0
    base_url = "https://api.osf.io/v2/preprints/"
    
    for query in queries:
        print(f"[EarthArXiv] '{query}'")
        try:
            params = {
                "filter[provider]": "eartharxiv",
                "filter[title,description]": query,
                "page[size]": 25,
            }
            r = requests.get(base_url, params=params, timeout=30)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}")
                continue
            data = r.json()
            papers = data.get("data", [])
            count = 0
            for p in papers:
                ea_id = p.get("id", "")
                if not ea_id or ea_id in seen_ids:
                    continue
                attrs = p.get("attributes", {})
                abstract = attrs.get("description", "")
                if not abstract or len(abstract) < 50:
                    continue
                seen_ids.add(ea_id)
                has_pdf = False
                pdf_name = f"ea_{ea_id}.pdf"
                download_url = f"https://eartharxiv.org/{ea_id}/download"
                if download_pdf(download_url, os.path.join(pdf_dir, pdf_name)):
                    has_pdf = True
                    pdfs_downloaded += 1
                time.sleep(1)
                all_papers.append({
                    "id": f"eartharxiv_{ea_id}", "source": "eartharxiv", "ea_id": ea_id,
                    "title": attrs.get("title", ""), "abstract": abstract,
                    "year": attrs.get("date_published", "")[:4] if attrs.get("date_published") else "",
                    "has_pdf": has_pdf,
                    "collected_at": datetime.now().isoformat()
                })
                count += 1
            print(f"  New: {count}, Total: {len(all_papers)}")
            time.sleep(1)
        except Exception as e:
            print(f"  Error: {e}")
    
    with open(os.path.join(output_dir, "eartharxiv_papers.json"), "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"\n[EarthArXiv] Total: {len(all_papers)}, PDFs: {pdfs_downloaded}")

def main():
    print("=" * 60)
    print("Finishing remaining downloads")
    print("=" * 60)
    
    print("\n--- ArXiv PDFs ---")
    download_remaining_arxiv()
    
    print("\n--- EarthArXiv ---")
    collect_eartharxiv()
    
    # Final summary
    import glob
    arxiv_pdfs = len(glob.glob("data/raw/arxiv/pdfs/*.pdf"))
    ea_pdfs = len(glob.glob("data/raw/eartharxiv/pdfs/*.pdf"))
    
    print("\n" + "=" * 60)
    print(f"FINAL PDF COUNT:")
    print(f"  ArXiv PDFs: {arxiv_pdfs}")
    print(f"  EarthArXiv PDFs: {ea_pdfs}")
    print(f"  Total PDFs: {arxiv_pdfs + ea_pdfs}")
    print("=" * 60)

if __name__ == "__main__":
    main()
