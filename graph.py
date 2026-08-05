"""
Cogni's core LangGraph flow.

Flow:
  user query
    -> router node (LLM classifies: "general_finance" vs "spending_question")
    -> if general_finance: RAG retrieval node -> answer node
    -> if spending_question: spending tool node -> answer node
    -> answer node produces the final natural-language response

This is intentionally a small graph (router + two branches + answer),
scoped to be genuinely understandable end-to-end rather than a large
multi-agent system I couldn't defend in an interview.
"""

import os
from typing import TypedDict, Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from rag import get_retriever
from tools import spending_summary, top_expenses

MODEL_NAME = "gemini-3.5-flash"  # free-tier model, generous daily rate limits


class CogniState(TypedDict):
    query: str
    route: Optional[Literal["general_finance", "spending_question"]]
    context: Optional[str]
    answer: Optional[str]


def get_llm(temperature: float = 0.2):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Export it before running the app, "
            "or add it to Streamlit secrets when deploying. "
            "Get a free key at aistudio.google.com."
        )
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=temperature, google_api_key=api_key)


ROUTER_SYSTEM_PROMPT = """You are a router for a personal finance assistant.
Classify the user's question into exactly one category:

- "spending_question": the user is asking about THEIR OWN spending, transactions,
  totals, categories, or biggest expenses (e.g. "how much did I spend on food?",
  "what are my top expenses?", "break down my spending by category").
- "general_finance": the user is asking a general finance/budgeting question that
  does not require their personal transaction data (e.g. "what is the 50/30/20 rule?",
  "how much should my emergency fund be?", "avalanche vs snowball method").

Respond with ONLY the category label, nothing else."""


def router_node(state: CogniState) -> CogniState:
    llm = get_llm(temperature=0)
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=state["query"]),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()
    if route not in ("general_finance", "spending_question"):
        route = "general_finance"  # safe default
    return {**state, "route": route}


def rag_node(state: CogniState) -> CogniState:
    retriever = get_retriever(k=3)
    docs = retriever.invoke(state["query"])
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return {**state, "context": context}


def spending_tool_node(state: CogniState) -> CogniState:
    # Simple heuristic tool selection. A larger system could let the LLM
    # choose args; kept explicit here so behavior stays predictable and
    # easy to explain.
    query_lower = state["query"].lower()
    if "top" in query_lower or "biggest" in query_lower or "largest" in query_lower:
        context = top_expenses(n=5)
    else:
        category = None
        for cat in ["food", "transport", "subscriptions", "utilities",
                    "entertainment", "rent", "housing", "savings", "investment"]:
            if cat in query_lower:
                category = cat
                break
        context = spending_summary(category=category)
    return {**state, "context": context}


ANSWER_SYSTEM_PROMPT = """You are Cogni, a helpful personal finance assistant.
Answer the user's question using ONLY the provided context. Be concise,
friendly, and concrete — use actual numbers from the context where relevant.
If the context doesn't fully answer the question, say what you can and note
what's missing. Do not invent figures not present in the context."""


def answer_node(state: CogniState) -> CogniState:
    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Question: {state['query']}\n\nContext:\n{state['context']}"
        ),
    ]
    response = llm.invoke(messages)
    return {**state, "answer": response.content}


def route_decision(state: CogniState) -> str:
    return state["route"]


def build_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(CogniState)
    graph.add_node("router", router_node)
    graph.add_node("rag_retrieval", rag_node)
    graph.add_node("spending_tool", spending_tool_node)
    graph.add_node("generate_answer", answer_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "general_finance": "rag_retrieval",
            "spending_question": "spending_tool",
        },
    )
    graph.add_edge("rag_retrieval", "generate_answer")
    graph.add_edge("spending_tool", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


def ask_cogni(query: str) -> str:
    app = build_graph()
    result = app.invoke({"query": query, "route": None, "context": None, "answer": None})
    return result["answer"]


if __name__ == "__main__":
    print(ask_cogni("What is the 50/30/20 rule?"))
    print("=====")
    print(ask_cogni("How much did I spend on food?"))
