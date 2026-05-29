"""
eda.py  –  Exploratory Data Analysis
=====================================
Standalone, self-contained EDA for the Dubai Real Estate dataset.
Produces: figures/eda_correlation_heatmap.png
          figures/eda_price_trends.png
          figures/eda_distributions.png
Run independently:  python src/eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns

# ── load data ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "dubai_realestate_2010_2025.csv")

def load():
    return pd.read_csv(DATA_PATH)

# ── 1. Descriptive Statistics ──────────────────────────────────────────────
def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["RPPI_AED_sqft","RRI_Index","CPI_Inflation_pct",
            "Fed_Funds_Rate_pct","FDI_USD_bn","Transactions_000s"]
    stats = df[cols].describe().T
    stats["skew"]  = df[cols].skew()
    stats["kurt"]  = df[cols].kurt()
    return stats

# ── 2. Correlation Heatmap ─────────────────────────────────────────────────
def correlation_heatmap(df: pd.DataFrame, out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)
    cols = ["RPPI_AED_sqft","CPI_Inflation_pct","Fed_Funds_Rate_pct",
            "FDI_USD_bn","Transactions_000s","Pop_Growth_pct"]
    labels = ["RPPI","CPI","FedRate","FDI","Transactions","PopGrowth"]
    corr   = df[cols].corr()
    corr.index = corr.columns = labels

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=.5,
                annot_kws={"size": 10}, ax=ax)
    ax.set_title("Pearson Correlation Matrix – Dubai RE Dataset (2010-2025)",
                 fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "eda_correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return corr

# ── 3. Price & Macro Time-Series ──────────────────────────────────────────
def price_trends(df: pd.DataFrame, out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    color_map = {"Bull": "#16a34a", "Bear": "#dc2626"}

    # Panel A – RPPI
    ax = axes[0]
    for i in range(len(df) - 1):
        c = color_map.get(df["Price_Trend"].iloc[i + 1], "#64748b")
        ax.fill_between(df["Year"].iloc[i:i+2],
                         df["RPPI_AED_sqft"].iloc[i:i+2], alpha=0.25, color=c)
        ax.plot(df["Year"].iloc[i:i+2],
                df["RPPI_AED_sqft"].iloc[i:i+2], color=c, lw=2)
    ax.set_ylabel("AED / sqft", fontsize=10)
    ax.set_title("Residential Property Price Index (AED/sqft)", fontweight="bold")
    ax.axhline(df["RPPI_AED_sqft"].mean(), ls="--", lw=1, color="#94a3b8",
               label=f"Mean = {df['RPPI_AED_sqft'].mean():.0f}")
    ax.legend(fontsize=9)

    # Panel B – CPI vs FedRate
    ax2 = axes[1]
    ax2.bar(df["Year"], df["CPI_Inflation_pct"],
            color="#f59e0b", alpha=0.7, label="CPI Inflation (%)")
    ax2b = ax2.twinx()
    ax2b.plot(df["Year"], df["Fed_Funds_Rate_pct"],
              color="#7c3aed", lw=2, marker="o", ms=4, label="Fed Funds Rate (%)")
    ax2.set_ylabel("CPI %", fontsize=10)
    ax2b.set_ylabel("Fed Rate %", fontsize=10, color="#7c3aed")
    ax2.set_title("Inflation (CPI) vs Global Monetary Conditions (Fed Rate)", fontweight="bold")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    # Panel C – FDI vs Transactions
    ax3 = axes[2]
    ax3.fill_between(df["Year"], df["FDI_USD_bn"],
                     alpha=0.4, color="#0ea5e9", label="FDI (USD bn)")
    ax3.set_ylabel("FDI USD bn", fontsize=10, color="#0ea5e9")
    ax3b = ax3.twinx()
    ax3b.plot(df["Year"], df["Transactions_000s"],
              color="#f97316", lw=2, marker="s", ms=4, label="Transactions (000s)")
    ax3b.set_ylabel("Transactions (000s)", fontsize=10, color="#f97316")
    ax3.set_title("Capital Flows (FDI) vs Market Activity (Transactions)", fontweight="bold")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax3.set_xlabel("Year", fontsize=10)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Dubai Real Estate – Macro Dashboard (2010-2025)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "eda_price_trends.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ── 4. Distribution Plots ─────────────────────────────────────────────────
def distributions(df: pd.DataFrame, out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)
    cols   = ["RPPI_AED_sqft","CPI_Inflation_pct","FDI_USD_bn",
              "Transactions_000s","Fed_Funds_Rate_pct","RRI_Index"]
    titles = ["RPPI (AED/sqft)","CPI Inflation (%)","FDI Inflows (USD bn)",
              "Transactions (000s)","Fed Funds Rate (%)","Rental Index"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for i, (col, title) in enumerate(zip(cols, titles)):
        ax = axes[i]
        sns.histplot(df[col], ax=ax, kde=True, color="#3b82f6",
                     edgecolor="white", alpha=0.7)
        ax.axvline(df[col].mean(), color="#ef4444", ls="--", lw=1.5,
                   label=f"Mean={df[col].mean():.2f}")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel("")
    plt.suptitle("Variable Distribution Analysis – Dubai RE (2010-2025)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "eda_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    df    = load()
    stats = descriptive(df)
    print("\n── Descriptive Statistics ──")
    print(stats.round(3))
    corr  = correlation_heatmap(df)
    print("\n── Correlation Matrix ──")
    print(corr.round(3))
    price_trends(df)
    distributions(df)
    print("\nEDA complete.")
