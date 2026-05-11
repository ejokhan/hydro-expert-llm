"""
Phase 1: Collect 50,000+ papers with full PDFs where available
Sources: PubMed, ArXiv (+PDFs), EarthArXiv (+PDFs), Semantic Scholar, USGS (+PDFs), NOAA, EPA
"""
import json
import os
import time
import requests
from datetime import datetime

def collect_pubmed(queries, max_per_query=1000, output_dir="data/raw/pubmed"):
    """PubMed: abstracts only (full text is paywalled)"""
    from Bio import Entrez, Medline
    Entrez.email = "ihaq@uvm.edu"
    os.makedirs(output_dir, exist_ok=True)
    existing_path = os.path.join(output_dir, "pubmed_papers.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        seen_ids = {p["pmid"] for p in existing}
        all_papers = existing
    else:
        seen_ids = set()
        all_papers = []
    for query in queries:
        print(f"[PubMed] '{query}'")
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_per_query, sort="relevance")
            results = Entrez.read(handle)
            handle.close()
            ids = results["IdList"]
            print(f"  Found {len(ids)}")
            for i in range(0, len(ids), 100):
                batch_ids = [pid for pid in ids[i:i+100] if pid not in seen_ids]
                if not batch_ids:
                    continue
                handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="text")
                records = list(Medline.parse(handle))
                handle.close()
                for record in records:
                    pmid = record.get("PMID", "")
                    if pmid in seen_ids:
                        continue
                    seen_ids.add(pmid)
                    abstract = record.get("AB", "")
                    if not abstract:
                        continue
                    all_papers.append({
                        "id": f"pubmed_{pmid}", "source": "pubmed", "pmid": pmid,
                        "title": record.get("TI", ""), "abstract": abstract,
                        "authors": record.get("AU", []), "journal": record.get("JT", ""),
                        "year": record.get("DP", "").split()[0] if record.get("DP") else "",
                        "keywords": record.get("MH", []),
                        "has_pdf": False,
                        "collected_at": datetime.now().isoformat()
                    })
                time.sleep(0.4)
        except Exception as e:
            print(f"  Error: {e}")
        print(f"  Total: {len(all_papers)}")
    with open(existing_path, "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"\n[PubMed] TOTAL: {len(all_papers)}")
    return all_papers

def download_pdf(url, save_path, timeout=30):
    """Download a PDF file. Returns True if successful."""
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False

def collect_arxiv(queries, max_per_query=300, output_dir="data/raw/arxiv"):
    """ArXiv: abstracts + FREE PDF downloads"""
    import arxiv
    os.makedirs(output_dir, exist_ok=True)
    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    existing_path = os.path.join(output_dir, "arxiv_papers.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        seen_ids = {p["arxiv_id"] for p in existing}
        all_papers = existing
    else:
        seen_ids = set()
        all_papers = []
    existing_pdfs = set(os.listdir(pdf_dir))
    pdfs_downloaded = 0
    pdfs_failed = 0
    for query in queries:
        print(f"[ArXiv] '{query}'")
        try:
            client = arxiv.Client()
            search = arxiv.Search(query=query, max_results=max_per_query, sort_by=arxiv.SortCriterion.Relevance)
            count = 0
            for result in client.results(search):
                arxiv_id = result.entry_id.split("/")[-1]
                if arxiv_id in seen_ids:
                    # Check if we need to download PDF for existing paper
                    pdf_name = f"{arxiv_id.replace('/', '_')}.pdf"
                    if pdf_name not in existing_pdfs:
                        if download_pdf(result.pdf_url, os.path.join(pdf_dir, pdf_name)):
                            pdfs_downloaded += 1
                            existing_pdfs.add(pdf_name)
                        else:
                            pdfs_failed += 1
                        time.sleep(1)
                    continue
                seen_ids.add(arxiv_id)
                # Download PDF
                pdf_name = f"{arxiv_id.replace('/', '_')}.pdf"
                pdf_path = os.path.join(pdf_dir, pdf_name)
                has_pdf = False
                if pdf_name not in existing_pdfs:
                    if download_pdf(result.pdf_url, pdf_path):
                        has_pdf = True
                        pdfs_downloaded += 1
                        existing_pdfs.add(pdf_name)
                    else:
                        pdfs_failed += 1
                    time.sleep(1)
                else:
                    has_pdf = True
                all_papers.append({
                    "id": f"arxiv_{arxiv_id}", "source": "arxiv", "arxiv_id": arxiv_id,
                    "title": result.title, "abstract": result.summary,
                    "authors": [a.name for a in result.authors],
                    "categories": result.categories,
                    "year": str(result.published.year),
                    "pdf_url": result.pdf_url,
                    "has_pdf": has_pdf,
                    "pdf_path": f"pdfs/{pdf_name}" if has_pdf else "",
                    "collected_at": datetime.now().isoformat()
                })
                count += 1
                if (pdfs_downloaded + pdfs_failed) % 50 == 0 and pdfs_downloaded > 0:
                    print(f"    PDFs: {pdfs_downloaded} downloaded, {pdfs_failed} failed")
            print(f"  New: {count}, Total: {len(all_papers)}")
        except Exception as e:
            print(f"  Error: {e}")
    with open(existing_path, "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"\n[ArXiv] TOTAL: {len(all_papers)} papers, {pdfs_downloaded} new PDFs downloaded")
    return all_papers

def collect_semantic_scholar(queries, max_per_query=200, output_dir="data/raw/semantic_scholar"):
    """Semantic Scholar: abstracts only (aggregates multiple sources)"""
    os.makedirs(output_dir, exist_ok=True)
    existing_path = os.path.join(output_dir, "semantic_scholar_papers.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        seen_ids = {p["ss_id"] for p in existing}
        all_papers = existing
    else:
        seen_ids = set()
        all_papers = []
    for query in queries:
        print(f"[Semantic Scholar] '{query}'")
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {"query": query, "limit": min(max_per_query, 100),
                     "fields": "title,abstract,year,authors,externalIds,openAccessPdf"}
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                print("  Rate limited, waiting 60s...")
                time.sleep(60)
                r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}")
                continue
            data = r.json()
            papers = data.get("data", [])
            count = 0
            for p in papers:
                ss_id = p.get("paperId", "")
                if not ss_id or ss_id in seen_ids:
                    continue
                abstract = p.get("abstract", "")
                if not abstract:
                    continue
                seen_ids.add(ss_id)
                all_papers.append({
                    "id": f"ss_{ss_id}", "source": "semantic_scholar", "ss_id": ss_id,
                    "title": p.get("title", ""), "abstract": abstract,
                    "authors": [a.get("name", "") for a in p.get("authors", [])],
                    "year": str(p.get("year", "")),
                    "has_pdf": False,
                    "collected_at": datetime.now().isoformat()
                })
                count += 1
            print(f"  New: {count}, Total: {len(all_papers)}")
            time.sleep(3)
        except Exception as e:
            print(f"  Error: {e}")
    with open(existing_path, "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"\n[Semantic Scholar] TOTAL: {len(all_papers)}")
    return all_papers

def collect_usgs(queries, max_per_query=200, output_dir="data/raw/usgs"):
    """USGS Publications: abstracts + FREE PDF downloads"""
    os.makedirs(output_dir, exist_ok=True)
    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    existing_path = os.path.join(output_dir, "usgs_papers.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        seen_ids = {p["usgs_id"] for p in existing}
        all_papers = existing
    else:
        seen_ids = set()
        all_papers = []
    base_url = "https://pubs.er.usgs.gov/pubs-services/publication"
    existing_pdfs = set(os.listdir(pdf_dir))
    pdfs_downloaded = 0
    for query in queries:
        print(f"[USGS] '{query}'")
        try:
            params = {"q": query, "pageSize": max_per_query}
            r = requests.get(base_url, params=params, timeout=30)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}")
                continue
            data = r.json()
            records = data.get("records", [])
            count = 0
            for record in records:
                usgs_id = str(record.get("id", ""))
                if usgs_id in seen_ids:
                    continue
                abstract = record.get("docAbstract", "")
                if not abstract:
                    continue
                seen_ids.add(usgs_id)
                # Try to find PDF link
                pdf_url = ""
                has_pdf = False
                links = record.get("links", [])
                for link in links:
                    link_url = link.get("url", "")
                    if link_url.endswith(".pdf"):
                        pdf_url = link_url
                        break
                    if "pdf" in link_url.lower():
                        pdf_url = link_url
                        break
                # Download PDF if available
                if pdf_url:
                    pdf_name = f"usgs_{usgs_id}.pdf"
                    if pdf_name not in existing_pdfs:
                        if download_pdf(pdf_url, os.path.join(pdf_dir, pdf_name)):
                            has_pdf = True
                            pdfs_downloaded += 1
                            existing_pdfs.add(pdf_name)
                        time.sleep(1)
                    else:
                        has_pdf = True
                all_papers.append({
                    "id": f"usgs_{usgs_id}", "source": "usgs", "usgs_id": usgs_id,
                    "title": record.get("title", ""), "abstract": abstract,
                    "authors": [a.get("text", "") for a in record.get("contributors", {}).get("authors", [])],
                    "year": str(record.get("publicationYear", "")),
                    "publication_type": record.get("publicationType", {}).get("text", ""),
                    "pdf_url": pdf_url,
                    "has_pdf": has_pdf,
                    "collected_at": datetime.now().isoformat()
                })
                count += 1
            print(f"  New: {count}, Total: {len(all_papers)}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error: {e}")
    with open(existing_path, "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"\n[USGS] TOTAL: {len(all_papers)} papers, {pdfs_downloaded} PDFs downloaded")
    return all_papers

def collect_eartharxiv(queries, max_per_query=200, output_dir="data/raw/eartharxiv"):
    """EarthArXiv: earth science preprints with FREE PDFs"""
    os.makedirs(output_dir, exist_ok=True)
    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    existing_path = os.path.join(output_dir, "eartharxiv_papers.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        seen_ids = {p["ea_id"] for p in existing}
        all_papers = existing
    else:
        seen_ids = set()
        all_papers = []
    existing_pdfs = set(os.listdir(pdf_dir))
    pdfs_downloaded = 0
    base_url = "https://api.osf.io/v2/preprints/"
    for query in queries:
        print(f"[EarthArXiv] '{query}'")
        try:
            params = {
                "filter[provider]": "eartharxiv",
                "filter[title,description]": query,
                "page[size]": min(max_per_query, 25),
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
                # Get PDF link
                pdf_url = ""
                has_pdf = False
                primary_file = p.get("links", {}).get("preprint_doi", "")
                # Try direct download link
                download_url = f"https://eartharxiv.org/{ea_id}/download"
                pdf_name = f"ea_{ea_id}.pdf"
                if pdf_name not in existing_pdfs:
                    if download_pdf(download_url, os.path.join(pdf_dir, pdf_name)):
                        has_pdf = True
                        pdfs_downloaded += 1
                        existing_pdfs.add(pdf_name)
                    time.sleep(1)
                else:
                    has_pdf = True
                all_papers.append({
                    "id": f"eartharxiv_{ea_id}", "source": "eartharxiv", "ea_id": ea_id,
                    "title": attrs.get("title", ""), "abstract": abstract,
                    "authors": [],
                    "year": attrs.get("date_published", "")[:4] if attrs.get("date_published") else "",
                    "has_pdf": has_pdf,
                    "collected_at": datetime.now().isoformat()
                })
                count += 1
            print(f"  New: {count}, Total: {len(all_papers)}")
            time.sleep(1)
        except Exception as e:
            print(f"  Error: {e}")
    with open(existing_path, "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"\n[EarthArXiv] TOTAL: {len(all_papers)} papers, {pdfs_downloaded} PDFs downloaded")
    return all_papers

def main():
    hydro_core = [
        "streamflow prediction machine learning",
        "hydrological modeling deep learning",
        "flood forecasting neural network",
        "rainfall runoff model deep learning",
        "groundwater level prediction machine learning",
        "river discharge time series forecasting",
        "watershed hydrology data driven",
        "evapotranspiration estimation machine learning",
        "snow melt runoff prediction",
        "drought prediction machine learning",
        "soil moisture prediction deep learning",
        "reservoir operation optimization",
        "sediment transport modeling",
        "baseflow separation method",
        "catchment hydrology transfer learning",
        "continental scale hydrological model",
        "physics informed neural network hydrology",
        "water demand forecasting deep learning",
        "hydrograph prediction LSTM",
        "urban stormwater management machine learning",
        "dam safety monitoring machine learning",
        "irrigation water management AI",
        "wetland hydrology modeling",
        "karst hydrology prediction",
        "glacier melt runoff forecasting",
    ]
    anomaly_queries = [
        "anomaly detection water quality sensor",
        "anomaly detection streamflow time series",
        "outlier detection hydrological data",
        "sensor malfunction detection water monitoring",
        "data quality control streamflow",
        "automated quality assurance water data",
        "spike detection environmental sensor",
        "drift detection water quality sensor",
        "fault detection water distribution network",
        "abnormal pattern detection river discharge",
        "time series anomaly detection environmental monitoring",
        "quality control hydrometric data",
        "erroneous data detection streamgaging",
        "sensor fault diagnosis water treatment",
        "real time anomaly detection water infrastructure",
        "ice jam detection river sensor",
        "backwater effect detection streamflow",
        "instrument error detection hydrology",
    ]
    water_quality = [
        "water quality prediction machine learning",
        "dissolved oxygen prediction neural network",
        "turbidity forecasting deep learning",
        "contaminant detection water supply",
        "water treatment optimization AI",
        "drinking water quality classification",
        "river water quality index prediction",
        "harmful algal bloom prediction machine learning",
        "eutrophication prediction",
        "heavy metal contamination water prediction",
    ]
    groundwater = [
        "groundwater modeling machine learning",
        "aquifer characterization deep learning",
        "well monitoring anomaly detection",
        "groundwater recharge estimation",
        "saltwater intrusion prediction",
        "groundwater contamination detection",
        "water table fluctuation prediction",
        "pumping test analysis machine learning",
    ]
    climate_water = [
        "climate change impact water resources",
        "extreme precipitation prediction",
        "flood frequency analysis climate change",
        "drought monitoring remote sensing",
        "snow water equivalent estimation",
        "evaporation prediction machine learning",
        "sea level rise coastal flooding",
        "monsoon prediction deep learning",
    ]
    remote_sensing = [
        "satellite remote sensing water quality",
        "MODIS streamflow estimation",
        "Landsat water body detection",
        "SAR flood mapping machine learning",
        "remote sensing soil moisture",
    ]
    time_series = [
        "time series forecasting transformer model",
        "time series anomaly detection deep learning",
        "time series foundation model pretraining",
        "temporal convolutional network forecasting",
        "self supervised learning time series",
        "transfer learning time series prediction",
        "multivariate time series anomaly detection",
        "attention mechanism time series",
    ]

    all_queries = hydro_core + anomaly_queries + water_quality + groundwater + climate_water + remote_sensing + time_series

    print("=" * 60)
    print("HYDRO-FINETUNE: Complete Paper Collection")
    print(f"Total queries: {len(all_queries)}")
    print(f"Sources: PubMed (abstracts), ArXiv (abstracts+PDFs),")
    print(f"  EarthArXiv (abstracts+PDFs), Semantic Scholar (abstracts),")
    print(f"  USGS (abstracts+PDFs)")
    print("=" * 60)

    # Source 1: PubMed (abstracts, no PDFs available)
    pubmed = collect_pubmed(all_queries, max_per_query=1000)

    # Source 2: ArXiv (abstracts + PDFs)
    arxiv_papers = collect_arxiv(all_queries, max_per_query=300)

    # Source 3: Semantic Scholar (abstracts, fills gaps)
    ss = collect_semantic_scholar(all_queries, max_per_query=100)

    # Source 4: USGS (abstracts + PDFs)
    usgs = collect_usgs(all_queries, max_per_query=200)

    # Source 5: EarthArXiv (earth science preprints + PDFs)
    eartharxiv = collect_eartharxiv(all_queries[:40], max_per_query=50)

    total = len(pubmed) + len(arxiv_papers) + len(ss) + len(usgs) + len(eartharxiv)

    # Count PDFs
    pdf_counts = {
        "arxiv": len([p for p in arxiv_papers if p.get("has_pdf")]),
        "usgs": len([p for p in usgs if p.get("has_pdf")]),
        "eartharxiv": len([p for p in eartharxiv if p.get("has_pdf")]),
    }

    summary = {
        "total_papers": total,
        "pubmed": len(pubmed),
        "arxiv": len(arxiv_papers),
        "semantic_scholar": len(ss),
        "usgs": len(usgs),
        "eartharxiv": len(eartharxiv),
        "total_pdfs": sum(pdf_counts.values()),
        "pdf_breakdown": pdf_counts,
        "collection_date": datetime.now().isoformat(),
        "num_queries": len(all_queries)
    }
    with open("data/collection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"TOTAL PAPERS: {total}")
    print(f"  PubMed:           {len(pubmed)} (abstracts only)")
    print(f"  ArXiv:            {len(arxiv_papers)} ({pdf_counts['arxiv']} with PDFs)")
    print(f"  Semantic Scholar:  {len(ss)} (abstracts only)")
    print(f"  USGS:             {len(usgs)} ({pdf_counts['usgs']} with PDFs)")
    print(f"  EarthArXiv:       {len(eartharxiv)} ({pdf_counts['eartharxiv']} with PDFs)")
    print(f"  TOTAL PDFs:       {sum(pdf_counts.values())}")
    print("=" * 60)

if __name__ == "__main__":
    main()
