"""
RAG pipeline for Cogni.

Loads the budgeting principles markdown doc, splits it into chunks,
embeds them locally with a sentence-transformers model, and builds
a FAISS vector store for retrieval. Embeddings run locally (no API
calls), so this part is free and fast to iterate on.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "budgeting_principles.md")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "faiss_index")

HEADERS_TO_SPLIT_ON = [
    ("#", "doc_title"),
    ("##", "section"),
]


def load_and_split_docs():
    with open(DATA_PATH, "r") as f:
        text = f.read()
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    chunks = splitter.split_text(text)
    return chunks


def get_embeddings():
    # Small, fast, local model — no external API calls needed for embeddings.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_or_load_vectorstore():
    embeddings = get_embeddings()
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    chunks = load_and_split_docs()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore


def get_retriever(k: int = 3):
    vectorstore = build_or_load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    # Quick manual test
    retriever = get_retriever()
    results = retriever.invoke("what is the 50/30/20 rule?")
    for r in results:
        print("---")
        print(r.metadata)
        print(r.page_content[:200])
