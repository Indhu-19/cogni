"""
Tools the agent can call for spending-related questions.
"""

import os
import pandas as pd

TRANSACTIONS_PATH = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")

# Map user keywords -> actual CSV category values
CATEGORY_MAP = {
    "food": "Food",
    "transport": "Transport",
    "subscriptions": "Subscriptions",
    "utilities": "Utilities",
    "entertainment": "Entertainment",
    "rent": "Rent/Housing",
    "housing": "Rent/Housing",
    "savings": "Savings/Investment",
    "investment": "Savings/Investment",
}


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])
    return df


def spending_summary(category: str | None = None, month: str | None = None) -> str:
    """
    Summarize spending, optionally filtered by category and/or month (YYYY-MM).
    """
    df = load_transactions()

    if month:
        df = df[df["date"].dt.strftime("%Y-%m") == month]

    if category:
        # Normalize keyword to real CSV category
        key = category.lower().strip()
        real_cat = CATEGORY_MAP.get(key, category)
        matched = df[df["category"].str.lower() == real_cat.lower()]
        if matched.empty:
            # also try partial match
            matched = df[df["category"].str.lower().str.contains(key, na=False)]
        if matched.empty:
            available = ", ".join(sorted(df["category"].unique()))
            return f"No transactions found for category '{category}'. Available categories: {available}"
        total = matched["amount"].sum()
        count = len(matched)
        return f"Category: {real_cat} | Total spent: ₹{total:,.0f} | Transactions: {count}"

    breakdown = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    total = df["amount"].sum()
    lines = [f"Total spending: ₹{total:,.0f}"]
    for cat, amt in breakdown.items():
        pct = (amt / total) * 100
        lines.append(f"  {cat}: ₹{amt:,.0f} ({pct:.1f}%)")
    return "\n".join(lines)


def top_expenses(n: int = 5) -> str:
    """Return the n largest individual transactions."""
    df = load_transactions()
    top = df.nlargest(n, "amount")
    lines = [
        f"{row.date.date()} | {row.description} | {row.category} | ₹{row.amount:,.0f}"
        for row in top.itertuples()
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(spending_summary())
    print("---")
    print(spending_summary(category="food"))
    print("---")
    print(spending_summary(category="rent"))
    print("---")
    print(top_expenses())
