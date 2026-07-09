"""Flask web API for querying the gofo agent."""

from flask import Flask, jsonify, request

import config
from core.models import QueryRequest
from tools.rag.retriever import retrieve

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/query")
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    top_k = int(data.get("n_results", config.TOP_K_DEFAULT))
    chunks = retrieve(QueryRequest(question=question, top_k=top_k))

    return jsonify(
        {
            "question": question,
            "answer": None,
            "sources": [chunk.model_dump() for chunk in chunks],
            "capability": "rag",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
