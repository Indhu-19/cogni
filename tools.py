"""
Tools the agent can call for spending-related questions.

This is the 'agentic' piece of Cogni: instead of only retrieving text
via RAG, the graph can route to a function that actually computes
something over the user's transaction data.
"""

import os
import pandas as pd

TRANSACTIONS_PATH = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])
    return df


def spending_summary(category: str | None = None, month: str | None = None) -> str:
    df = load_transactions()

    if month:
        df = df[df["date"].dt.strftime("%Y-%m") == month]

    if category:
        matched = df[df["category"].str.lower() == category.lower()]
        if matched.empty:
            available = ", ".join(sorted(df["category"].unique()))
            return f"No transactions found for category '{category}'. Available categories: {available}"
        total = matched["amount"].sum()
        count = len(matched)
        return f"Category: {category} | Total spent: ₹{total} | Transactions: {count}"

    breakdown = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    total = df["amount"].sum()
    lines = [f"Total spending: ₹{total}"]
    for cat, amt in breakdown.items():
        pct = (amt / total) * 100
        lines.append(f"  {cat}: ₹{amt} ({pct:.1f}%)")
    return "\n".join(lines)


def top_expenses(n: int = 5) -> str:
    df = load_transactions()
    top = df.nlargest(n, "amount")
    lines = [f"{row.date.date()} | {row.description} | {row.category} | ₹{row.amount}" for row in top.itertuples()]
    return "\n".join(lines)


if __name__ == "__main__":
    print(spending_summary())
    print("---")
    print(spending_summary(category="Food"))
    print("---")
    print(top_expenses())
