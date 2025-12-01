from typing import List, Dict
from flask import Flask, render_template, request, send_from_directory

from src.search_engine import search_tfidf, search_embeddings
from src.config import RAW_SLIDES_DIR

app = Flask(__name__)

@app.route("/slides/<path:filename>")
def serve_slide(filename: str):
    return send_from_directory(RAW_SLIDES_DIR, filename)

def _run_search(query: str, mode: str, top_k: int,) -> Dict[str, List[Dict]]:
    results_tfidf: List[Dict] = []
    results_emb: List[Dict] = []

    if not query.strip():
        return {"tfidf": results_tfidf, "emb": results_emb}

    if mode in ("tfidf", "both"):
        results_tfidf = search_tfidf(query, top_k=top_k)

    if mode in ("emb", "both"):
        results_emb = search_embeddings(query, top_k=top_k)

    return {"tfidf": results_tfidf, "emb": results_emb}

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    mode = "both"
    top_k = 5
    results_tfidf: List[Dict] = []
    results_emb: List[Dict] = []

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        mode = request.form.get("mode", "both")

        try:
            top_k = int(request.form.get("top_k", "5"))
        except ValueError:
            top_k = 5

        search_results = _run_search(query, mode, top_k)
        results_tfidf = search_results["tfidf"]
        results_emb = search_results["emb"]

    return render_template("index.html", query=query, mode=mode, top_k=top_k, results_tfidf=results_tfidf, results_emb=results_emb)

if __name__ == "__main__":
    app.run(debug=True)