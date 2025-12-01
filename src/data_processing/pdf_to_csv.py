import os
import csv
import fitz 

from pathlib import Path

from src.config import RAW_SLIDES_DIR, SLIDES_CORPUS_CSV

def extract_page_text(page) -> str:
    text = page.get_text("text")

    if not text:
        return ""
    
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return " ".join(lines)

def build_pdf_index(input_dir: str | Path, output_csv: str | Path):
    input_dir = Path(input_dir)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    rows = []

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem 
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc, start=1):
            text = extract_page_text(page)
            text = text.strip()

            if not text:
                continue  

            rows.append(
                {
                    "doc_id": doc_id,
                    "page_number": page_num,
                    "text_raw": text,
                }
            )

    output_csv = Path(output_csv)
    os.makedirs(output_csv.parent, exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "page_number", "text_raw"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    build_pdf_index(input_dir=RAW_SLIDES_DIR, output_csv=SLIDES_CORPUS_CSV)