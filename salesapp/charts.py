"""Matplotlib charts. Light surface, one-hue sequential ramps for magnitude,
fixed-order categorical hues for identity, thin marks, no dual axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRID,
    }
)


def inr(x: float, pos=None) -> str:
    a = abs(x)
    if a >= 1e7:
        return f"₹{x / 1e7:.1f} Cr"
    if a >= 1e5:
        return f"₹{x / 1e5:.1f} L"
    if a >= 1e3:
        return f"₹{x / 1e3:.0f}K"
    return f"₹{x:.0f}"


def _fig(title: str, subtitle: str = "", size=(11, 5.2)):
    fig, ax = plt.subplots(figsize=size)
    fig.text(0.015, 0.97, title, fontsize=14, fontweight="bold", va="top", ha="left")
    if subtitle:
        fig.text(0.015, 0.915, subtitle, fontsize=10, color=INK2, va="top", ha="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, out_dir: Path, name: str, left: float = 0.08, top: float = 0.84) -> Path:
    path = Path(out_dir) / name
    fig.subplots_adjust(top=top, left=left, right=0.97, bottom=0.1)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _heat(ax, data: np.ndarray, fmt, vmax: float | None = None):
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=SEQ, aspect="auto", vmin=0, vmax=vmax or np.nanmax(data))
    norm = im.norm
    for (i, j), v in np.ndenumerate(data):
        if np.isnan(v):
            continue
        ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8,
                color="white" if norm(v) > 0.55 else INK)
    ax.set_xticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def revenue_trend(monthly: pd.DataFrame, forecast: pd.DataFrame, out_dir: Path) -> Path:
    last = monthly["revenue"].iloc[-1]
    fig, ax = _fig("Monthly revenue and forecast",
                   f"Delivered orders, net of discounts. Latest month {inr(last)}.")
    ax.plot(monthly.index, monthly["revenue"], color=BLUE, lw=2, label="Actual")
    fx = [monthly.index[-1], *forecast.index]
    ax.plot(fx, [last, *forecast["forecast"]], color=BLUE, lw=2, ls=(0, (4, 3)), label="Forecast")
    ax.fill_between(fx, [last, *forecast["lower"]], [last, *forecast["upper"]], color=BLUE, alpha=0.14, lw=0)
    ax.scatter([forecast.index[-1]], [forecast["forecast"].iloc[-1]], s=40, color=BLUE,
               edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.yaxis.set_major_formatter(FuncFormatter(inr))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, out_dir, "01_revenue_trend.png")


def category_region_heatmap(pivot: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = _fig("Revenue by category and region", "Delivered net revenue; darker means more.", (10, 5))
    _heat(ax, pivot.to_numpy(float), lambda v: inr(v))
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.xaxis.tick_top()
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.tick_params(length=0)
    return _save(fig, out_dir, "02_category_region_heatmap.png", left=0.15, top=0.80)


def pareto_chart(par: dict, out_dir: Path) -> Path:
    prod, cust, th = par["products"], par["customers"], par["threshold"]
    fig, ax = _fig("How concentrated is revenue?",
                   f"{par['customer_share_for_threshold']:.0%} of customers and "
                   f"{par['product_share_for_threshold']:.0%} of products generate {th:.0%} of revenue.")
    ax.plot(np.r_[0, prod["share_of_items"]], np.r_[0, prod["cum_revenue_share"]], color=BLUE, lw=2, label="Products")
    ax.plot(np.r_[0, cust["share_of_items"]], np.r_[0, cust["cum_revenue_share"]], color=ORANGE, lw=2, label="Customers")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax.axhline(th, color=MUTED, lw=1, ls=":")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of products / customers (largest first)")
    ax.set_ylabel("Cumulative share of revenue")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, out_dir, "03_pareto_concentration.png")


def cohort_heatmap(retention: pd.DataFrame, sizes: pd.Series, out_dir: Path, max_age: int = 12) -> Path:
    data = retention.iloc[:, : max_age + 1].to_numpy(float) * 100
    fig, ax = _fig("Customer retention by signup cohort",
                   "Share of each first-order cohort that orders again N months later.", (11, 7.5))
    _heat(ax, data, lambda v: f"{v:.0f}", vmax=60)
    ax.set_xticks(range(data.shape[1]), [str(i) for i in range(data.shape[1])])
    ax.xaxis.tick_top()
    ax.set_xlabel("Months since first order")
    ax.xaxis.set_label_position("top")
    labels = [f"{d:%b %y}  (n={sizes.loc[d]})" for d in retention.index]
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.tick_params(length=0)
    return _save(fig, out_dir, "04_cohort_retention.png", left=0.12, top=0.83)


def rfm_chart(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.text(0.015, 0.97, "RFM customer segments", fontsize=14, fontweight="bold", va="top")
    top_seg = summary["revenue"].idxmax()
    row = summary.loc[top_seg]
    fig.text(0.015, 0.915,
             f"{top_seg} are {row['customer_share_pct']:.0f}% of customers but {row['revenue_share_pct']:.0f}% of revenue.",
             fontsize=10, color=INK2, va="top")
    order = summary.sort_values("revenue").index
    for ax, col, title, fmt in (
        (a1, "customers", "Customers", lambda v: f"{v:,.0f}"),
        (a2, "revenue", "Revenue", inr),
    ):
        vals = summary.loc[order, col]
        ax.barh(vals.index, vals.to_numpy(), color=BLUE, height=0.6)
        for y, v in enumerate(vals.to_numpy()):
            ax.text(v, y, " " + fmt(v), va="center", fontsize=9, color=INK2)
        ax.set_title(title, loc="left", fontsize=10, color=INK2)
        ax.set_xticks([])
        ax.margins(x=0.22)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(top=0.82, left=0.16, right=0.97, bottom=0.05, wspace=0.08)
    path = Path(out_dir) / "05_rfm_segments.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def anomaly_chart(daily: pd.DataFrame, out_dir: Path) -> Path:
    flagged = daily[daily["is_anomaly"]]
    fig, ax = _fig("Daily revenue anomalies",
                   f"{len(flagged)} days flagged by robust z-score (median / MAD on a trailing 28-day baseline).")
    ax.plot(daily.index, daily["revenue"], color=BLUE, lw=0.9, label="Daily revenue")
    ax.plot(daily.index, daily["baseline"], color=MUTED, lw=1.4, ls=(0, (4, 3)), label="28-day median")
    ax.scatter(flagged.index, flagged["revenue"], s=48, color=ORANGE, edgecolor=SURFACE, linewidth=2,
               zorder=3, label="Anomaly")
    for d, r in flagged.iterrows():
        ax.annotate(f"{d:%d %b %Y}\n{r['uplift_pct']:+.0f}% vs baseline", (d, r["revenue"]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.yaxis.set_major_formatter(FuncFormatter(inr))
    ax.set_ylim(0, daily["revenue"].max() * 1.22)
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, out_dir, "06_anomalies.png")


def make_all(monthly, forecast, pivot, pareto, retention, sizes, rfm_summary, daily, out_dir) -> list[Path]:
    out = Path(out_dir) / "charts"
    out.mkdir(parents=True, exist_ok=True)
    return [
        revenue_trend(monthly, forecast, out),
        category_region_heatmap(pivot, out),
        pareto_chart(pareto, out),
        cohort_heatmap(retention, sizes, out),
        rfm_chart(rfm_summary, out),
        anomaly_chart(daily, out),
    ]
