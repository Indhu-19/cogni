"""
RAG pipeline for Cogni.
Loads budgeting principles, splits by markdown headers, embeds with Gemini, stores in FAISS.
"""

import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "budgeting_principles.md")

HEADERS_TO_SPLIT_ON = [
    ("#", "doc_title"),
    ("##", "section"),
]


def load_and_split_docs():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise RuntimeError(f"Knowledge base is empty: {DATA_PATH}")
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    chunks = splitter.split_text(text)
    return chunks


def get_embeddings():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Export it before running the app, "
            "or add it to Streamlit secrets when deploying."
        )
    # Widely supported embedding model
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )


_VECTORSTORE_CACHE = None


def build_or_load_vectorstore():
    global _VECTORSTORE_CACHE
    if _VECTORSTORE_CACHE is not None:
        return _VECTORSTORE_CACHE
    embeddings = get_embeddings()
    chunks = load_and_split_docs()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    _VECTORSTORE_CACHE = vectorstore
    return vectorstore


def get_retriever(k: int = 3):
    vectorstore = build_or_load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    retriever = get_retriever()
    results = retriever.invoke("what is the 50/30/20 rule?")
    for r in results:
        print("---")
        print(r.metadata)
        print(r.page_content[:300])
