#!/usr/bin/env python3
"""Sales analytics CLI.

    python main.py generate            # create data/raw_sales.csv (messy synthetic data)
    python main.py run                 # clean, analyse, chart, write output/report.md
    python main.py all                 # generate + run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from salesapp import analytics as A
from salesapp import charts
from salesapp.cleaning import load_and_clean
from salesapp.data_generator import generate_sales_data, write_sales_csv
from salesapp.report import write_report

log = logging.getLogger("sales")


def cmd_generate(args: argparse.Namespace) -> None:
    df = generate_sales_data(n_customers=args.customers, seed=args.seed)
    path = write_sales_csv(df, args.data)
    log.info("wrote %s rows to %s", f"{len(df):,}", path)


def cmd_run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    df, rep = load_and_clean(args.data)
    log.info("cleaned %s -> %s rows (%.1f -> %.1f MB)", f"{rep.rows_in:,}", f"{rep.rows_out:,}",
             rep.mem_before_mb, rep.mem_after_mb)

    orders = A.build_orders(df)
    kpis = A.kpi_summary(df, orders)
    monthly = A.monthly_trend(orders)
    forecast, finfo = A.forecast_revenue(monthly["revenue"], horizon=args.horizon)
    cat = A.category_summary(df)
    pivot = A.category_region_pivot(df)
    products = A.top_products(df)
    channel = A.channel_mix(orders)
    par = A.pareto_analysis(df)
    retention, sizes = A.cohort_retention(orders)
    rfm = A.rfm_table(orders)
    rfm_summary = A.rfm_segment_summary(rfm)
    anomalies = A.detect_anomalies(orders, threshold=args.z_threshold)
    pairs = A.basket_pairs(df)

    for name, frame in {
        "monthly_trend": monthly, "forecast": forecast, "category_summary": cat,
        "category_region_pivot": pivot, "top_products": products, "channel_mix": channel,
        "cohort_retention": retention, "customers_rfm": rfm, "rfm_segments": rfm_summary,
        "anomalies": anomalies[anomalies["is_anomaly"]], "basket_pairs": pairs,
        "cleaning_log": rep.to_frame(),
    }.items():
        frame.to_csv(tables / f"{name}.csv")
    (tables / "kpis.json").write_text(json.dumps(kpis, indent=2), encoding="utf-8")
    df.to_parquet(out / "clean_sales.parquet") if args.parquet else df.to_csv(out / "clean_sales.csv", index=False)

    charts.make_all(monthly, forecast, pivot, par, retention, sizes, rfm_summary, anomalies, out)
    report = write_report(
        out / "report.md", kpis=kpis, rep=rep, monthly=monthly, forecast=forecast, finfo=finfo, cat=cat,
        products=products, par=par, retention=retention, rfm_summary=rfm_summary, anomalies=anomalies,
        pairs=pairs, channel=channel,
    )
    log.info("report written to %s", report)
    print(f"\nNet revenue: {charts.inr(kpis['Net revenue (delivered)'])} | "
          f"AOV: {charts.inr(kpis['Average order value'])} | "
          f"Repeat rate: {kpis['Repeat customer rate (%)']:.1f}% | "
          f"Anomalous days: {int(anomalies['is_anomaly'].sum())}")
    print(f"Open {report} (charts in {out / 'charts'})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Advanced pandas sales analytics")
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn, help_ in (("generate", cmd_generate, "create messy synthetic data"),
                            ("run", cmd_run, "clean data and build the report"),
                            ("all", None, "generate then run")):
        s = sub.add_parser(name, help=help_)
        s.set_defaults(fn=fn)
        s.add_argument("--data", default="data/raw_sales.csv", help="raw CSV path")
        s.add_argument("--out", default="output", help="output directory")
        s.add_argument("--customers", type=int, default=3000)
        s.add_argument("--seed", type=int, default=42)
        s.add_argument("--horizon", type=int, default=3, help="forecast months")
        s.add_argument("--z-threshold", type=float, default=8.0, help="anomaly robust-z cutoff")
        s.add_argument("--parquet", action="store_true", help="save cleaned data as parquet (needs pyarrow)")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    if args.command == "all":
        cmd_generate(args)
        cmd_run(args)
    else:
        args.fn(args)


if __name__ == "__main__":
    main()
