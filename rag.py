"""
RAG pipeline for Cogni.

Loads the budgeting principles markdown doc, splits it into chunks,
embeds them using Gemini's embedding model, and builds a FAISS vector
store for retrieval. Uses the same GEMINI_API_KEY as the LLM calls,
so no separate model download or credential is needed.
"""

import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Export it before running the app, "
            "or add it to Streamlit secrets when deploying."
        )
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)


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
