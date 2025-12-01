import re

from typing import List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer  

from src.config import URL_PATTERN, MULTISPACE_PATTERN, WORD_PATTERN, URL_PATTERN

stemmer = PorterStemmer()
STOPWORDS = set(ENGLISH_STOP_WORDS)

def clean_for_tfidf(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    tokens: List[str] = WORD_PATTERN.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)

def clean_for_embeddings(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()
    
    if not text:
        return ""

    text = URL_PATTERN.sub(" ", text)
    text = text.lower()
    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    return text