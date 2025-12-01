import re

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_SLIDES_DIR: Path = DATA_DIR / "raw_slides"
PROCESSED_DIR: Path = DATA_DIR / "processed"

SLIDES_CORPUS_CSV: Path = PROCESSED_DIR / "slides_corpus_raw.csv"
SLIDES_CORPUS_TFIDF: Path = PROCESSED_DIR / "slides_corpus_tfidf.csv"
SLIDES_CORPUS_EMBED: Path = PROCESSED_DIR / "slides_corpus_embed.csv"
FINAL_SLIDES_COROUS: Path = PROCESSED_DIR / "final_slides_corpus.csv"

WORD_PATTERN = re.compile(r"\w+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_PATTERN = re.compile(r"\s+")

MODELS_DIR: Path = PROJECT_ROOT / "models"

TFIDF_DIR: Path = MODELS_DIR / "tfidf"
TFIDF_VECTORIZER_PATH: Path = TFIDF_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH: Path = TFIDF_DIR / "tfidf_matrix.pkl"

EMBEDDINGS_DIR: Path = MODELS_DIR / "embeddings"
EMBEDDINGS_NPY_PATH: Path = EMBEDDINGS_DIR / "embeddings.npy"
EMBEDDINGS_METADATA_PATH: Path = EMBEDDINGS_DIR / "metadata.pkl"

TEMPLATES_DIR: Path = PROJECT_ROOT / "templates"
STATIC_DIR: Path = PROJECT_ROOT / "static"

SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

TFIDF_MAX_FEATURES: int = 20000
TFIDF_NGRAM_RANGE = (1, 2) 
TFIDF_MIN_DF: int = 1
TFIDF_MAX_DF: float = 0.9

DEFAULT_TOP_K: int = 5

MIN_COSINE_SIMILARITY: float = 0.0

LOWERCASE_TEXT: bool = True
REMOVE_PUNCTUATION: bool = True
REMOVE_NUMBERS: bool = True

def ensure_directories() -> None:
    for directory in [
        DATA_DIR,
        RAW_SLIDES_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        TFIDF_DIR,
        EMBEDDINGS_DIR,
        TEMPLATES_DIR,
        STATIC_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()