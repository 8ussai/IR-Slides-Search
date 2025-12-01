import os

import pandas as pd

from src.config import SLIDES_CORPUS_CSV, SLIDES_CORPUS_TFIDF, SLIDES_CORPUS_EMBED, FINAL_SLIDES_COROUS
from src.data_processing.text_cleaning import clean_for_tfidf, clean_for_embeddings

def build_tfidf_corpus(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    df["text_tfidf"] = df["text_raw"].astype(str).apply(clean_for_tfidf)
    df = df[df["text_tfidf"].str.len() > 0]
    df.to_csv(output_csv, index=False)

def build_embeddings_corpus(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    df["text_embed"] = df["text_raw"].astype(str).apply(clean_for_embeddings)
    df = df[df["text_embed"].str.len() > 0]
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    build_tfidf_corpus(input_csv=SLIDES_CORPUS_CSV, output_csv=SLIDES_CORPUS_TFIDF)
    build_embeddings_corpus(input_csv=SLIDES_CORPUS_CSV, output_csv=SLIDES_CORPUS_EMBED)

    df1 = pd.read_csv(SLIDES_CORPUS_TFIDF)
    df2 = pd.read_csv(SLIDES_CORPUS_EMBED)
    df1["text_embed"] = df2["text_embed"]
    df1.to_csv(FINAL_SLIDES_COROUS, index=False)
    
    os.remove(SLIDES_CORPUS_CSV)
    os.remove(SLIDES_CORPUS_TFIDF)
    os.remove(SLIDES_CORPUS_EMBED)