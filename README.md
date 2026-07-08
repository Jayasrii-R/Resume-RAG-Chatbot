📌 Project Overview

Resume RAG Chatbot is an AI-powered application that allows users to upload their resume and ask questions about it using **Retrieval-Augmented Generation (RAG)**.

The application extracts information from resumes, stores embeddings in a vector database, retrieves relevant context, and generates intelligent responses using **Google Gemini**.

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Vector Databases
- Semantic Search
- Prompt Engineering
✨ Features

- 📄 Upload Resume (PDF)
- 🤖 Chat with your Resume
- 🔍 Semantic Search using Embeddings
- 📚 Chroma Vector Database
- ⚡ Google Gemini API Integration
- 💬 Context-aware AI Responses
- 🎨 Interactive Streamlit Interface
- 🔒 Secure API Key Management
🛠 Tech Stack

| Category | Technologies |
|-----------|-------------|
| Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini |
| Framework | LangChain |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| Embeddings | Google Embeddings |
| Environment | Python Virtual Environment |
📂 Project Structure

```
Resume-RAG-Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── data/
│
├── vectorstore/
│
├── assets/
│   ├── home.png
│   ├── upload.png
│   ├── chatbot.png
│
└── utils/
 🧠 How RAG Works
 PDF Resume
                     │
                     ▼
             Text Extraction
                     │
                     ▼
              Text Chunking
                     │
                     ▼
         Google Embeddings API
                     │
                     ▼
            Chroma Vector Store
                     │
         Similarity Search
                     │
                     ▼
             Relevant Context
                     │
                     ▼
             Google Gemini LLM
                     │
                     ▼
              Intelligent Answer


💡 Example Questions

- Tell me about my skills.
- Summarize my experience.
- What programming languages do I know?
- Which projects have I completed?
- What certifications are listed?
- What are my strengths?
- Give interview questions based on my resume.

📈 Future Improvements

- Multiple Resume Support
- Voice Interaction
- Resume Scoring
- ATS Compatibility Checker
- Job Description Matching
- Authentication
- Chat History
- Cloud Deployment

🎯 Learning Outcomes

- Retrieval-Augmented Generation (RAG)
- Large Language Models
- Prompt Engineering
- LangChain
- Google Gemini API
- ChromaDB
- Semantic Search
- Streamlit Deployment
- Vector Embeddings
