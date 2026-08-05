# Cogni — AI-Powered Personal Finance Companion

Cogni is a conversational finance assistant that answers two kinds of questions:

1. **General finance/budgeting questions** (e.g. *"what is the 50/30/20 rule?"*)
   — answered via **RAG retrieval** over a curated budgeting-principles knowledge base.
2. **Personal spending questions** (e.g. *"how much did I spend on food this month?"*)
   — answered via a **tool call** that computes real numbers from a transactions dataset.

A **LangGraph router** classifies each incoming query and directs it down the
right path before a final answer node synthesizes a natural-language response.

## Why this architecture

Rather than a single monolithic RAG pipeline, Cogni routes between retrieval
and computation because these are genuinely different tasks: "what is the
50/30/20 rule" needs *knowledge retrieval*, while "how much did I spend on
food" needs *deterministic computation over structured data* — an LLM
shouldn't be guessing arithmetic when a `pandas` groupby can just compute it
correctly. This is a small step toward agentic behavior: the system decides
*which capability to invoke* rather than always doing the same thing.

## Architecture

```
User Query
    │
    ▼
┌─────────┐
│ Router  │  (LLM classifies: general_finance vs spending_question)
└────┬────┘
     │
 ┌───┴────┐
 ▼         ▼
RAG      Spending
Retrieval  Tool
 │         │
 └────┬────┘
      ▼
  ┌────────┐
  │ Answer │  (LLM synthesizes final response from retrieved context)
  └────────┘
```

## Tech Stack

- **LangGraph** — orchestrates the router → retrieval/tool → answer flow
- **LangChain + FAISS** — vector store for RAG retrieval
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, no API cost
- **Google Gemini (`gemini-2.5-flash`)** — free-tier LLM for routing and answer generation
- **Streamlit** — frontend/UI
- **pandas** — transaction data analysis

## Project Structure

```
cogni/
├── app.py              # Streamlit frontend
├── graph.py             # LangGraph flow: router, RAG node, tool node, answer node
├── rag.py                # RAG pipeline: doc loading, chunking, FAISS vector store
├── tools.py              # Spending analysis functions (the agent's "tool")
├── data/
│   ├── budgeting_principles.md   # Knowledge base for RAG
│   └── transactions.csv          # Synthetic transaction data
└── requirements.txt
```

## Running Locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud. `GEMINI_API_KEY` is set via
Streamlit Secrets rather than committed to the repo.

## Known Limitations / Next Steps

This is a scoped MVP built to demonstrate the core RAG + agentic-routing
pattern end-to-end, rather than a full production system. Honest scope
limitations:

- Transaction data is synthetic, not connected to a real bank/UPI feed.
- Category extraction in the spending tool uses keyword matching rather
  than LLM-based entity extraction — reliable for demo queries, but would
  need generalizing for open-ended phrasing.
- No conversation memory across turns yet (each query is independent).
- No user authentication — single-user demo.

Planned extensions: multi-turn memory via LangGraph checkpointing, LLM-based
tool-argument extraction instead of keyword matching, and real transaction
import (CSV upload).
