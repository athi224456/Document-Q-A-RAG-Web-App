"""
app.py
------
Flask web application for the RAG document Q&A system.

Pages:
    GET  /                  -> dashboard (upload documents, see index stats)
    POST /upload             -> ingest one or more uploaded files
    POST /documents/<id>/remove -> remove a document from the index
    GET  /chat               -> chat interface
    POST /chat                -> ask a question, get an answer + sources
    POST /chat/reset          -> clear conversation history

JSON API:
    GET  /health              -> {"status": "ok"}
    POST /api/ask             -> {"question": "..."} -> answer + sources

Run:
    python src/app.py
Then open:
    http://localhost:5001
"""
import os

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

import config
from rag_pipeline import RAGPipeline

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

pipeline = RAGPipeline()

# In-memory conversation history (single-user demo app; resets on restart)
_chat_history = []


def _allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS


# ---------- Dashboard ----------

@app.route("/", methods=["GET"])
def dashboard():
    return render_template(
        "dashboard.html",
        active="dashboard",
        documents=pipeline.documents,
        num_documents=pipeline.num_documents,
        num_chunks=pipeline.num_chunks,
        provider=config.LLM_PROVIDER,
        active_provider=config.LLM_PROVIDER,
    )


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        flash("No file selected.", "error")
        return redirect(url_for("dashboard"))

    indexed, skipped = [], []
    for f in files:
        if f.filename == "":
            continue
        if not _allowed_file(f.filename):
            skipped.append(f.filename)
            continue
        filename = secure_filename(f.filename)
        save_path = os.path.join(config.UPLOAD_DIR, filename)
        f.save(save_path)
        try:
            pipeline.ingest_file(save_path, filename)
            indexed.append(filename)
        except Exception as e:
            skipped.append(f"{filename} ({e})")

    if indexed:
        flash(f"Indexed {len(indexed)} document(s): {', '.join(indexed)}", "ok")
    if skipped:
        flash(f"Skipped: {', '.join(skipped)}", "error")

    return redirect(url_for("dashboard"))


@app.route("/documents/<doc_id>/remove", methods=["POST"])
def remove_document(doc_id):
    pipeline.remove_document(doc_id)
    flash("Document removed from the index.", "ok")
    return redirect(url_for("dashboard"))


# ---------- Chat ----------

@app.route("/chat", methods=["GET"])
def chat():
    return render_template(
        "chat.html",
        active="chat",
        has_documents=pipeline.num_documents > 0,
        history=_chat_history,
        active_provider=config.LLM_PROVIDER,
    )


@app.route("/chat", methods=["POST"])
def chat_ask():
    question = request.form.get("question", "").strip()
    if question and pipeline.num_documents > 0:
        result = pipeline.answer(question)
        _chat_history.append(result)
    return redirect(url_for("chat"))


@app.route("/chat/reset", methods=["POST"])
def reset_chat():
    _chat_history.clear()
    return redirect(url_for("chat"))


# ---------- JSON API ----------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "documents": pipeline.num_documents, "chunks": pipeline.num_chunks})


@app.route("/api/ask", methods=["POST"])
def api_ask():
    payload = request.get_json(force=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "field 'question' is required"}), 400
    if pipeline.num_documents == 0:
        return jsonify({"error": "no documents have been indexed yet"}), 400
    result = pipeline.answer(question)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
