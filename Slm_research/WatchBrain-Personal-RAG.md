# EchoMem – Personal Memory Assistant for Smartwatch

**A private, voice-first, hallucination-free RAG system that turns your smartwatch into your second brain.**

You speak → it remembers **only what you tell it** → you ask later → it answers **exactly** from your own data.

No Google login, no cloud, no hallucinations, fully per-user (User 01, User 02 …).

---

## Project Goal (Final-Year / Portfolio Project)

- Store personal details, projects, CGPA, daily notes, etc. via voice.
- Retrieve them faithfully when asked.
- Multi-user isolation (simple User ID).
- Runs locally on phone + Wear OS companion (or prototype on laptop).

---

## High-Level Workflow

![RAG Pipeline Diagram](https://miro.medium.com/1*ayogAmoUrJ907xozhHIJ6w.png)  
(Source: Medium – RAG Pipeline)

![Simple RAG Flow](https://i.imgur.com/vACLc.png)  
(Source: Designveloper – Chunk → Embed → Vector Store → LLM)

![Smartwatch Personal AI Hub](https://media.licdn.com/dms/image/v2/D4D22AQE0v2iE0fGf_A/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1725... )  
(Source: LinkedIn – Personal AI Hub with Smartwatch)

---

## Detailed Data Storage Process (Ingestion Pipeline)

After user speaks:

1. **Audio → Text** → Whisper (local)
2. **Intent check** → “Store” or “Query”
3. **Chunking** (200–400 tokens, 50-token overlap)
4. **Embedding** → `all-MiniLM-L6-v2` (384-dim)
5. **Store in ChromaDB** (one collection per user)

### ChromaDB Structure

![ChromaDB Hierarchy](https://miro.medium.com/v2/resize:fit:1400/1*... )  
(Source: Medium – Chroma DB Introduction)

![Chroma Flow](https://www.projectpro.io/article/chromadb/1044)  
(Source: ProjectPro – Chroma Architecture)

![RAG process](https://medium.com/@chuciche/understanding-the-entire-process-of-retrieval-augmented-generation-rag-0d15a7f75b68)
(Source: Medium - Understanding the Entire Process of Retrieval-Augmented Generation (RAG) Step by Step)

**What gets saved (example row)**

| id | text (chunk)                                      | embedding (vector)          | metadata                                      |
|----|---------------------------------------------------|-----------------------------|-----------------------------------------------|
| 1  | I am a final-year student at IIT Bombay…         | [0.12, -0.45, …]            | {"user_id": "01", "timestamp": "2025-02-07", "original_text": "..."} |

Data is **persistent** on disk/phone (`./memory_db/chroma.sqlite3`).

---

## Retrieval + Faithfulness (No Hallucination Guarantee)

1. Query → embed
2. Search **only** that user’s collection
3. Relevance threshold ≥ 0.65
4. Strict prompt:

```text
You are a faithful personal memory assistant.
Answer ONLY using the context below.
If the answer is not in the context, reply exactly: "I don't know."
```
---

## Tech Stack (100% Free & Local)

1.Voice → faster-whisper / Whisper-large-v3

2.Embeddings → sentence-transformers/all-MiniLM-L6-v2

3.Vector DB → ChromaDB (persistent, multi-collection)

4.LLM → Ollama (Llama-3.1-8B / Phi-3-mini)

5.Framework → LangChain / LlamaIndex

6.Frontend → Streamlit (prototype) or Wear OS + Flutter companion

7.TTS → Piper / gTTS