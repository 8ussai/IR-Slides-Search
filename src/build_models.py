import pickle
import joblib

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    FINAL_SLIDES_COROUS,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    SENTENCE_TRANSFORMER_MODEL,
    EMBEDDINGS_NPY_PATH,
    EMBEDDINGS_METADATA_PATH,
)

def build_tfidf_index() -> None:
    df = pd.read_csv(FINAL_SLIDES_COROUS)
    texts = df["text_tfidf"].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)

def build_embeddings_index() -> None:
    df = pd.read_csv(FINAL_SLIDES_COROUS)
    texts = df["text_embed"].astype(str).tolist()
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype("float32")

    np.save(EMBEDDINGS_NPY_PATH, embeddings)

    metadata = list(
        zip(
            df["doc_id"].tolist(),
            df["page_number"].astype(int).tolist(),
        )
    )
    with open(EMBEDDINGS_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":
    build_tfidf_index()
    build_embeddings_index()