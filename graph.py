"""
Cogni LangGraph flow:
query -> router -> (RAG | spending tool) -> answer
"""

import os
from typing import TypedDict, Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from rag import get_retriever
from tools import spending_summary, top_expenses

MODEL_NAME = "gemini-3.5-flash-lite"


class CogniState(TypedDict):
    query: str
    route: Optional[Literal["general_finance", "spending_question"]]
    context: Optional[str]
    answer: Optional[str]


def get_llm(temperature: float = 0.2):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=temperature, google_api_key=api_key)


ROUTER_SYSTEM_PROMPT = """You are a router for a personal finance assistant.
Classify the user's question into exactly one category:

- "spending_question": the user is asking about THEIR OWN spending, transactions,
  totals, categories, or biggest expenses.
- "general_finance": the user is asking a general finance/budgeting question that
  does not require their personal transaction data.

Respond with ONLY the category label, nothing else."""


def router_node(state: CogniState) -> CogniState:
    llm = get_llm(temperature=0)
    response = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=state["query"]),
    ])
    route = response.content.strip().lower()
    if route not in ("general_finance", "spending_question"):
        route = "general_finance"
    return {**state, "route": route}


def rag_node(state: CogniState) -> CogniState:
    retriever = get_retriever(k=3)
    docs = retriever.invoke(state["query"])
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return {**state, "context": context}


def spending_tool_node(state: CogniState) -> CogniState:
    query_lower = state["query"].lower()
    if any(w in query_lower for w in ("top", "biggest", "largest")):
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
friendly, and concrete — use actual numbers from the context where relevant."""


def answer_node(state: CogniState) -> CogniState:
    llm = get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {state['query']}\n\nContext:\n{state['context']}"),
    ])
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
