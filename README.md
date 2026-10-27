Document Q&A — RAG Web App
A retrieval-augmented generation (RAG) web application: upload your own
documents, then ask questions about them in a chat interface. Every
answer shows exactly which passages it came from.
Works fully offline out of the box (no API key, no model downloads) using
TF-IDF retrieval and an extractive answer generator. Optionally, plug in
OpenAI or Anthropic for more fluent, synthesized answers.
How it works
```
Upload           Chunk           Index            Ask              Answer
document  ---->  into ~220-  --> with TF-IDF --> a question -->  the most relevant
(.pdf/.docx/     word pieces      + cosine                       chunks are retrieved,
.txt/.md)                         similarity                     then turned into an
                                                                  answer + shown sources
```
document_loader.py — extracts plain text from PDF, DOCX, TXT, or MD.
chunking.py — splits text into overlapping ~220-word chunks (so answers
can cite a specific passage instead of a whole document), and splits text
into sentences for the extractive generator.
retriever.py — builds a TF-IDF index over all chunks and ranks them
against a question with cosine similarity. No embedding model to
download, so it works immediately, offline.
generator.py — turns retrieved chunks into an answer.
extractive (default): picks and assembles the most relevant
sentences from the retrieved chunks. Zero setup.
openai / anthropic: sends the retrieved chunks as context to a
real LLM for a more natural, synthesized answer. Requires an API key.
If the call fails for any reason, the app automatically falls back to
the extractive answer rather than showing an error.
rag_pipeline.py — wires ingestion and Q&A together, and persists the
index to disk so it survives a restart.
app.py — the Flask web app.
Project structure
```
doc_qa_rag/
├── data/
│   ├── uploads/           # saved copies of uploaded files
│   └── index/             # persisted TF-IDF index (index.joblib)
├── src/
│   ├── config.py          # paths, chunking size, provider settings
│   ├── document_loader.py # PDF/DOCX/TXT/MD -> plain text
│   ├── chunking.py        # text -> chunks, text -> sentences
│   ├── retriever.py       # TF-IDF index + search
│   ├── generator.py       # extractive answer + optional LLM backends
│   ├── rag_pipeline.py    # ingestion + Q&A orchestration
│   ├── app.py             # Flask web app
│   ├── templates/         # dashboard.html, chat.html, base.html
│   └── static/css/        # styling
└── requirements.txt
```
Setup
```bash
cd doc_qa_rag
pip install -r requirements.txt
```
Run
```bash
cd src
python app.py
```
Open http://localhost:5001 in your browser.
Dashboard (`/`) — upload documents (PDF, DOCX, TXT, MD, up to 25 MB
each) and see what's currently indexed.
Chat (`/chat`) — ask questions. Each answer shows an expandable
"sources" section with the filename, match score, and the exact passage
used.
Using a real LLM instead of the extractive default
By default no API key is needed — the app answers using only the text of
your documents. To get more fluent answers from an LLM, set an environment
variable before running the app:
```bash
# OpenAI
export DOC_QA_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# or Anthropic
export DOC_QA_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...

python app.py
```
The LLM only ever sees the chunks retrieved for that specific question —
never your whole document library — which keeps prompts small and answers
grounded. If the API call fails (bad key, no network, rate limit), the app
falls back to the extractive answer automatically and marks it as a
fallback in the chat UI.
JSON API
For calling the pipeline from another service:
```bash
curl -X POST http://localhost:5001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund window?"}'
```
```bash
curl http://localhost:5001/health
```
Notes and limitations
Conversation history is kept in memory and resets when the server
restarts. The document index, however, is persisted to
`data/index/index.joblib` and survives restarts.
This is a single-user demo app (no accounts or authentication). Don't
expose it directly on the public internet without adding that.
TF-IDF retrieval matches on wording, not deep semantic meaning — it works
well for direct questions about content that's actually in the document,
but won't catch every paraphrase. Swapping in a dense embedding model
(e.g. sentence-transformers) later only requires changes inside
`retriever.py`; nothing else in the pipeline needs to change.
The extractive generator quotes/assembles your own text rather than
writing new sentences — that's intentional, so answers can never contain
information that isn't actually in your documents.
