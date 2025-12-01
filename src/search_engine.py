import re
import pickle
import joblib

import numpy as np
import pandas as pd

from typing import List, Dict
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer

from src.config import (
    FINAL_SLIDES_COROUS,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    EMBEDDINGS_NPY_PATH,
    EMBEDDINGS_METADATA_PATH,
    SENTENCE_TRANSFORMER_MODEL,
    DEFAULT_TOP_K,
    LOWERCASE_TEXT,
    REMOVE_PUNCTUATION,
    REMOVE_NUMBERS,
)

def clean_text(text: str) -> str:
    if LOWERCASE_TEXT:
        text = text.lower()

    if REMOVE_PUNCTUATION:
        text = re.sub(r"[^\w\s]", " ", text)

    if REMOVE_NUMBERS:
        text = re.sub(r"\d+", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_tfidf_index():
    df = pd.read_csv(FINAL_SLIDES_COROUS)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)

    return df, vectorizer, tfidf_matrix

def search_tfidf(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    df, vectorizer, tfidf_matrix = load_tfidf_index()
    cleaned_query = clean_text(query)

    if not cleaned_query:
        return []

    query_vec = vectorizer.transform([cleaned_query])
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()

    if top_k <= 0:
        top_k = DEFAULT_TOP_K

    top_indices = np.argsort(cosine_similarities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = df.iloc[idx]
        results.append(
            {
                "rank": rank,
                "score": float(cosine_similarities[idx]),
                "doc_id": row["doc_id"],
                "page_number": int(row["page_number"]),
                "text": row["text_raw"],
            }
        )
    return results

_df_cache = None
_embeddings_cache = None
_metadata_cache = None
_model_cache = None

def load_embeddings_index():
    global _df_cache, _embeddings_cache, _metadata_cache, _model_cache

    if _df_cache is None:
        _df_cache = pd.read_csv(FINAL_SLIDES_COROUS)

    if _embeddings_cache is None or _metadata_cache is None:
        _embeddings_cache = np.load(EMBEDDINGS_NPY_PATH)
        with open(EMBEDDINGS_METADATA_PATH, "rb") as f:
            _metadata_cache = pickle.load(f)

    if _model_cache is None:
        _model_cache = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    return _df_cache, _embeddings_cache, _metadata_cache, _model_cache


def cosine_similarity(query_vec: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    q = query_vec.reshape(1, -1)
    sims = np.dot(doc_embeddings, q.T).flatten()
    return sims


def search_embeddings(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    df, embeddings, metadata, model = load_embeddings_index()
    cleaned_query = clean_text(query)

    if not cleaned_query:
        return []

    query_vec = model.encode(
        [cleaned_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    sims = cosine_similarity(query_vec, embeddings)

    if top_k <= 0:
        top_k = DEFAULT_TOP_K

    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        doc_id, page_number = metadata[idx]
        row = df.iloc[idx]

        results.append(
            {
                "rank": rank,
                "score": float(sims[idx]),
                "doc_id": doc_id,
                "page_number": int(page_number),
                "text": row["text_raw"],
            }
        )
    return results

def interactive_cli():
    print("=== Combined Search Engine (TF-IDF + Embeddings) ===")
    print("Type your query (or just press Enter to exit).")
    print("--------------------------------------------------")

    while True:
        query = input("\nQuery: ").strip()
        if not query:
            print("Exiting...")
            break

        print("\n--- TF-IDF RESULTS ---")
        for r in search_tfidf(query, 5):
            print(f"[{r['rank']}] {r['doc_id']} (p{r['page_number']}) - {r['score']:.4f}")

        print("\n--- EMBEDDINGS RESULTS ---")
        for r in search_embeddings(query, 5):
            print(f"[{r['rank']}] {r['doc_id']} (p{r['page_number']}) - {r['score']:.4f}")


if __name__ == "__main__":
    interactive_cli()
