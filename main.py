"""Synent Data Science Internship - Task 5: Sales Data Analysis (Superstore)
Author: Hassan Butt
Run: python main.py
"""

import sys, os, re, json, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


DATASETS = {'superstore': {'encoding': 'latin-1',
                'files': ['Sample - Superstore.csv', 'SampleSuperstore.csv', 'Superstore.csv'],
                'kaggle': 'vivek468/superstore-dataset-final',
                'name': 'Superstore Sales dataset',
                'page': 'https://www.kaggle.com/datasets/vivek468/superstore-dataset-final'}}



def _outdir(repo_dir):
    p = Path(repo_dir) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, repo_dir, name):
    fig.savefig(_outdir(repo_dir) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def save_json(obj, repo_dir, name):
    (_outdir(repo_dir) / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _read_csv(path, spec):
    kw = {"encoding": spec.get("encoding", "utf-8")}
    if spec.get("no_header_file") and Path(path).name == spec["no_header_file"]:
        kw["header"] = None
        kw["names"] = spec["names"]
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def _builtin(key):
    if key == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame.drop(columns=["target"])
        df["species"] = [d.target_names[i] for i in d.target]
        return df
    raise KeyError(key)


def get_data(key, repo_dir):
    """Load a dataset: local data/ folder -> built-in -> kagglehub -> public URL."""
    spec = DATASETS[key]
    data_dir = Path(repo_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        p = data_dir / name
        if p.exists():
            print(f"  [data] using local file {p}")
            return _read_csv(p, spec)
    if spec.get("builtin"):
        print("  [data] using built-in copy")
        return _builtin(key)
    if spec.get("kaggle"):
        try:
            import kagglehub
            folder = Path(kagglehub.dataset_download(spec["kaggle"]))
            cands = list(folder.rglob("*.csv"))
            pick = None
            for want in spec["files"]:
                for c in cands:
                    if c.name.lower() == want.lower():
                        pick = c
                        break
                if pick:
                    break
            if pick is None and cands:
                pick = cands[0]
            if pick is not None:
                dest = data_dir / pick.name
                shutil.copy(pick, dest)
                print(f"  [data] downloaded from Kaggle: {dest}")
                return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] kagglehub download not available ({type(e).__name__}: {e})")
    if spec.get("url"):
        try:
            import urllib.request
            dest = data_dir / (spec.get("no_header_file") or spec["files"][0])
            urllib.request.urlretrieve(spec["url"], dest)
            print(f"  [data] downloaded from public URL: {dest}")
            return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] public URL download failed ({type(e).__name__}: {e})")
    raise FileNotFoundError(
        f"Dataset '{spec['name']}' not found.\n"
        f"  -> Download it from {spec['page']}\n"
        f"  -> Save the CSV as one of {spec['files']} inside: {data_dir.resolve()}\n"
        f"  -> Then run the script again.")


def _to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def _snake(col):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()


def task5_run(repo_dir):
    df = get_data("superstore", repo_dir)
    df.columns = [_snake(c) for c in df.columns]
    shape = [int(df.shape[0]), int(df.shape[1])]
    df["order_date"] = _to_datetime(df["order_date"])
    df = df.dropna(subset=["order_date"])
    monthly = df.set_index("order_date")[["sales", "profit"]].resample("MS").sum()
    monthly["margin_pct"] = monthly["profit"] / monthly["sales"] * 100
    monthly.round(2).to_csv(_outdir(repo_dir) / "monthly_revenue.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly["sales"], marker="o", label="Sales (revenue)", color="#2980b9")
    ax.plot(monthly.index, monthly["profit"], marker="s", label="Profit", color="#27ae60")
    ax.set_title("Monthly revenue and profit trend")
    ax.legend()
    save_fig(fig, repo_dir, "monthly_revenue_trend.png")

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    season = df.groupby("month")["sales"].sum()
    yearly = df.groupby("year")[["sales", "profit"]].sum()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    season.plot(kind="bar", ax=ax[0], color="#e67e22")
    ax[0].set_title("Total sales by calendar month (seasonality)")
    ax[0].set_xlabel("Month")
    yearly.plot(kind="bar", ax=ax[1])
    ax[1].set_title("Sales and profit by year")
    ax[1].tick_params(axis="x", rotation=0)
    save_fig(fig, repo_dir, "seasonality_and_yearly.png")

    top_sales = df.groupby("product_name")["sales"].sum().sort_values(ascending=False).head(10)
    top_qty = df.groupby("product_name")["quantity"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    top_sales.sort_values().plot(kind="barh", ax=ax[0], color="#2980b9")
    ax[0].set_title("Top 10 products by revenue")
    top_qty.sort_values().plot(kind="barh", ax=ax[1], color="#8e44ad")
    ax[1].set_title("Top 10 products by quantity sold")
    fig.tight_layout()
    save_fig(fig, repo_dir, "top_selling_products.png")
    top_sales.round(2).to_csv(_outdir(repo_dir) / "top_products_by_revenue.csv")

    cat = df.groupby("category")[["sales", "profit"]].sum()
    cat["margin_pct"] = cat["profit"] / cat["sales"] * 100
    sub = df.groupby("sub_category")[["sales", "profit"]].sum()
    sub["margin_pct"] = sub["profit"] / sub["sales"] * 100
    sub = sub.sort_values("profit")
    reg = df.groupby("region")[["sales", "profit"]].sum()
    reg["margin_pct"] = reg["profit"] / reg["sales"] * 100
    sub.round(2).to_csv(_outdir(repo_dir) / "profit_by_sub_category.csv")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in sub["profit"]]
    ax[0].barh(sub.index, sub["profit"], color=colors)
    ax[0].set_title("Profit by sub-category (red = loss)")
    reg["margin_pct"].plot(kind="bar", ax=ax[1], color="#16a085")
    ax[1].set_title("Profit margin by region (%)")
    ax[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save_fig(fig, repo_dir, "profit_analysis.png")

    disc_effect = None
    if "discount" in df.columns:
        bands = pd.cut(df["discount"], bins=[-0.001, 0, 0.1, 0.2, 0.3, 0.5, 1.0],
                       labels=["0%", "1-10%", "11-20%", "21-30%", "31-50%", ">50%"])
        disc_effect = df.groupby(bands, observed=True).apply(lambda g: g["profit"].sum() / g["sales"].sum() * 100)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        disc_effect.plot(kind="bar", ax=ax, color="#d35400")
        ax.set_title("Profit margin (%) by discount band")
        ax.tick_params(axis="x", rotation=0)
        save_fig(fig, repo_dir, "discount_vs_profit.png")

    total_sales = float(df["sales"].sum())
    total_profit = float(df["profit"].sum())
    best_month = monthly["sales"].idxmax()
    worst_month = monthly["sales"].idxmin()
    loss_subs = sub[sub["profit"] < 0].index.tolist()
    best_sub = sub["profit"].idxmax()
    top_product = top_sales.index[0]
    peak_cal_month = int(season.idxmax())
    insights = [
        f"Total revenue was ${total_sales:,.0f} with ${total_profit:,.0f} profit (margin {total_profit / total_sales * 100:.1f}%).",
        f"Revenue peaked in {best_month:%B %Y} and was lowest in {worst_month:%B %Y}; calendar month {peak_cal_month} is the strongest overall.",
        f"The top product by revenue is '{top_product}' (${top_sales.iloc[0]:,.0f}).",
        (f"Loss-making sub-categories: {', '.join(loss_subs)}." if loss_subs else "No sub-category made a loss.")
        + f" The most profitable sub-category is {best_sub}.",
    ]
    if disc_effect is not None and len(disc_effect) > 1:
        trend_word = "fall" if disc_effect.iloc[0] > disc_effect.iloc[-1] else "do not fall"
        insights.append(f"Profit margins {trend_word} as discounts rise: {disc_effect.iloc[0]:.1f}% with no discount vs {disc_effect.iloc[-1]:.1f}% in the highest discount band.")
    report = ["# Business Insights Report", "", f"*Dataset: {shape[0]} order lines.*", ""]
    report += [f"- {i}" for i in insights]
    report += ["", "## Suggested actions (data-driven, to be validated by the business)",
               "- Review pricing and discount policy for the loss-making sub-categories.",
               "- Prepare inventory and marketing for the strongest calendar months.",
               "- Protect the top-selling products from stock-outs."]
    (_outdir(repo_dir) / "business_insights_report.md").write_text("\n".join(report), encoding="utf-8")
    return {"headline": {"Total revenue": f"${total_sales:,.0f}", "Total profit": f"${total_profit:,.0f}",
                         "Profit margin": f"{total_profit / total_sales * 100:.1f}%",
                         "Best month": f"{best_month:%b %Y}", "Top product": top_product},
            "insights": insights, "dataset_shape": shape,
            "files": ["monthly_revenue_trend.png", "seasonality_and_yearly.png", "top_selling_products.png",
                      "profit_analysis.png", "discount_vs_profit.png", "business_insights_report.md"]}



if __name__ == "__main__":
    res = task5_run(Path(__file__).parent)
    print(json.dumps(res, indent=2, default=str))
