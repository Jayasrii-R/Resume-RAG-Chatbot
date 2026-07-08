# Resume RAG

A Streamlit app that:
- Lets you upload one or more resume PDFs and indexes them with FAISS + sentence-transformer embeddings.
- Answers questions about the uploaded resumes (RAG, powered by Groq via LangChain).
- Ranks all uploaded resumes against a pasted job description, with strengths/gaps per candidate.

## Setup (using your existing venv)

You already have the dependencies installed based on your pip logs. Just add the two new files:

1. Copy `app.py` into `C:\Users\Jayas\OneDrive\Desktop\Resume-RAG\`
2. Copy `.env.example` to `.env` in the same folder and fill in your real key:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
   ```
   Get a free key at https://console.groq.com/keys
3. If anything is missing, from an activated venv:
   ```
   pip install -r requirements.txt
   ```

## Run it

```
cd C:\Users\Jayas\OneDrive\Desktop\Resume-RAG
venv\Scripts\activate
streamlit run app.py
```

This opens the app in your browser (usually http://localhost:8501).

## Using it

1. **Sidebar** — upload one or more resume PDFs, click "Process Resumes".
2. **Ask Questions tab** — ask things like "Who has 3+ years of Python experience?" and get an answer with source resume citations.
3. **Match to Job Description tab** — paste a JD, click "Rank Resumes" to get each candidate scored 0-100 with strengths/gaps.

## Notes

- Embeddings run locally (`sentence-transformers/all-MiniLM-L6-v2`), so no API needed for indexing/search — only the LLM calls (Q&A answers, JD scoring) use Groq.
- If `GROQ_MODEL` in `.env` ever 404s, check https://console.groq.com/docs/models for current available model names and update it.
- Everything resets when you restart Streamlit (in-memory FAISS index) — re-upload and reprocess resumes each session.
