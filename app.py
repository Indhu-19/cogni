import os
import streamlit as st
import traceback

# Streamlit Cloud: put GEMINI_API_KEY in the app's Secrets, it lands in env vars.
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

from graph import ask_cogni  # noqa: E402  (import after env var is set)

st.set_page_config(page_title="Cogni — AI Finance Companion", page_icon="💰")

st.title("💰 Cogni")
st.caption("An AI-powered personal finance companion — LangGraph + RAG + Gemini")

with st.expander("About this project"):
    st.markdown(
        """
        **Cogni** answers two kinds of questions:
        1. **General finance questions** (e.g. "what is the 50/30/20 rule?")
           — answered via RAG retrieval over a budgeting-principles knowledge base.
        2. **Personal spending questions** (e.g. "how much did I spend on food?")
           — answered via a tool that computes real numbers from a transactions dataset.

        A LangGraph router decides which path to take for each query.
        Built with LangGraph, FAISS, and Gemini (both for chat and embeddings).
        """
    )

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

sample_questions = [
    "What is the 50/30/20 rule?",
    "How much did I spend on food?",
    "What are my top 5 expenses?",
    "Give me a full breakdown of my spending by category.",
    "What's the difference between the avalanche and snowball debt methods?",
]

st.write("Try:")
cols = st.columns(len(sample_questions))
clicked_query = None
for col, q in zip(cols, sample_questions):
    if col.button(q, use_container_width=True):
        clicked_query = q

user_query = st.chat_input("Ask Cogni about budgeting or your spending...")
query = clicked_query if clicked_query is not None else user_query

if query is not None:

    query = query.strip()

    if query == "":
        st.stop()

    st.write("DEBUG Query:", repr(query))

    st.session_state.history.append(("user", query))

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                answer = ask_cogni(query)

            except Exception:
                answer = traceback.format_exc()

            st.markdown(answer)

    st.session_state.history.append(("assistant", answer))
