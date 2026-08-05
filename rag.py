"""
RAG pipeline for Cogni.
Loads budgeting principles, splits by markdown headers,
embeds LOCALLY with sentence-transformers, stores in FAISS.
"""
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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
    return splitter.split_text(text)

_EMBEDDINGS_CACHE = None

def get_embeddings():
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        # Runs on CPU, no API key, no network call at inference time
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDINGS_CACHE

_VECTORSTORE_CACHE = None

def build_or_load_vectorstore():
    global _VECTORSTORE_CACHE
    if _VECTORSTORE_CACHE is not None:
        return _VECTORSTORE_CACHE
    embeddings = get_embeddings()
    chunks = load_and_split_docs()
    _VECTORSTORE_CACHE = FAISS.from_documents(chunks, embeddings)
    return _VECTORSTORE_CACHE

def get_retriever(k: int = 3):
    return build_or_load_vectorstore().as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    retriever = get_retriever()
    for r in retriever.invoke("what is the 50/30/20 rule?"):
        print("---")
        print(r.metadata)
        print(r.page_content[:300])
