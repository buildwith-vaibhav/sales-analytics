"""Interactive Altair charts for the Streamlit dashboard (hover tooltips, zoom on time series)."""

from __future__ import annotations

import altair as alt
import pandas as pd

from .charts import inr

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]

_INR_LABEL = (
    "datum.value >= 10000000 ? '₹' + format(datum.value / 10000000, '.1f') + ' Cr'"
    " : datum.value >= 100000 ? '₹' + format(datum.value / 100000, '.1f') + ' L'"
    " : '₹' + format(datum.value, ',.0f')"
)


def _inr_axis(title: str | None = None) -> alt.Axis:
    return alt.Axis(labelExpr=_INR_LABEL, title=title, grid=True)


def _style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(labelColor=INK2, titleColor=INK2, gridColor=GRID, domainColor=GRID, tickColor=GRID)
        .configure_legend(labelColor=INK2, titleColor=INK2, orient="top", title=None)
        .configure_title(color=INK)
    )


def trend_chart(monthly: pd.DataFrame, forecast: pd.DataFrame | None) -> alt.TopLevelMixin:
    actual = monthly.reset_index()[["month", "revenue"]].assign(label=lambda d: d["revenue"].map(inr))
    x = alt.X("month:T", title=None, axis=alt.Axis(format="%b %y", grid=False))
    y = alt.Y("revenue:Q", axis=_inr_axis(), scale=alt.Scale(zero=True))
    tip = [alt.Tooltip("month:T", title="Month", format="%b %Y"), alt.Tooltip("label:N", title="Revenue")]

    layers = [
        alt.Chart(actual).mark_line(color=BLUE, strokeWidth=2).encode(x=x, y=y),
        alt.Chart(actual).mark_circle(size=90, opacity=0).encode(x=x, y=y, tooltip=tip),
    ]
    if forecast is not None and len(forecast):
        last = actual.iloc[-1]
        fc = forecast.reset_index()
        join = pd.DataFrame({"month": [last["month"]], "forecast": [last["revenue"]],
                             "lower": [last["revenue"]], "upper": [last["revenue"]]})
        path = pd.concat([join, fc], ignore_index=True)
        fc = fc.assign(label=fc["forecast"].map(inr),
                       range=fc["lower"].map(inr) + " to " + fc["upper"].map(inr))
        fx = alt.X("month:T", title=None)
        layers += [
            alt.Chart(path).mark_area(color=BLUE, opacity=0.14).encode(x=fx, y="lower:Q", y2="upper:Q"),
            alt.Chart(path).mark_line(color=BLUE, strokeWidth=2, strokeDash=[5, 4]).encode(x=fx, y="forecast:Q"),
            alt.Chart(fc).mark_circle(size=70, color=BLUE).encode(
                x=fx, y="forecast:Q",
                tooltip=[alt.Tooltip("month:T", title="Forecast for", format="%b %Y"),
                         alt.Tooltip("label:N", title="Forecast"), alt.Tooltip("range:N", title="95% range")]),
        ]
    return _style(alt.layer(*layers).properties(height=340))


def category_bar(cat: pd.DataFrame) -> alt.TopLevelMixin:
    d = cat.reset_index().assign(label=lambda t: t["revenue"].map(inr))
    base = alt.Chart(d).encode(
        y=alt.Y("category:N", sort="-x", title=None),
        x=alt.X("revenue:Q", axis=None),
    )
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, size=22).encode(
        tooltip=[alt.Tooltip("category:N", title="Category"), alt.Tooltip("label:N", title="Revenue"),
                 alt.Tooltip("share_pct:Q", title="Share %", format=".1f"),
                 alt.Tooltip("units:Q", title="Units", format=",")])
    text = base.mark_text(align="left", dx=5, color=INK2).encode(text="label:N")
    return _style((bars + text).properties(height=230))


def region_heatmap(pivot: pd.DataFrame) -> alt.TopLevelMixin:
    p = pivot.copy()
    p.columns, p.index = p.columns.astype(str), p.index.astype(str)
    p.index.name = "category"
    long = p.reset_index().melt(id_vars="category", var_name="region", value_name="revenue")
    long["label"] = long["revenue"].map(inr)
    cut = long["revenue"].max() * 0.55
    base = alt.Chart(long).encode(
        x=alt.X("region:N", sort=list(p.columns), title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("category:N", sort=list(p.index), title=None),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("revenue:Q", scale=alt.Scale(range=SEQ), legend=None),
        tooltip=[alt.Tooltip("category:N"), alt.Tooltip("region:N"), alt.Tooltip("label:N", title="Revenue")])
    text = base.mark_text(fontSize=11).encode(
        text="label:N",
        color=alt.condition(alt.datum.revenue > cut, alt.value("white"), alt.value(INK)))
    return _style((rect + text).properties(height=300))


def pareto_chart(par: dict) -> alt.TopLevelMixin:
    parts = []
    for name, key in (("Products", "products"), ("Customers", "customers")):
        c = par[key]
        parts.append(pd.DataFrame({
            "share": [0.0, *c["share_of_items"]],
            "revenue_share": [0.0, *c["cum_revenue_share"]],
            "series": name,
        }))
    d = pd.concat(parts, ignore_index=True)
    lines = alt.Chart(d).mark_line(strokeWidth=2).encode(
        x=alt.X("share:Q", title="Share of products / customers (largest first)", axis=alt.Axis(format="%")),
        y=alt.Y("revenue_share:Q", title="Cumulative share of revenue", axis=alt.Axis(format="%")),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Products", "Customers"], range=[BLUE, ORANGE])),
        tooltip=[alt.Tooltip("series:N"), alt.Tooltip("share:Q", title="Top share", format=".0%"),
                 alt.Tooltip("revenue_share:Q", title="Revenue share", format=".0%")])
    rule = alt.Chart(pd.DataFrame({"y": [par["threshold"]]})).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(y="y:Q")
    return _style((lines + rule).properties(height=320))


def rfm_chart(summary: pd.DataFrame) -> alt.TopLevelMixin:
    d = summary.reset_index()
    long = pd.concat([
        pd.DataFrame({"segment": d["segment"], "share": d["customer_share_pct"], "metric": "% of customers"}),
        pd.DataFrame({"segment": d["segment"], "share": d["revenue_share_pct"], "metric": "% of revenue"}),
    ])
    return _style(
        alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("segment:N", sort=list(d["segment"]), title=None),
            yOffset="metric:N",
            x=alt.X("share:Q", title="Share (%)"),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["% of customers", "% of revenue"], range=[BLUE, ORANGE])),
            tooltip=[alt.Tooltip("segment:N"), alt.Tooltip("metric:N"), alt.Tooltip("share:Q", format=".1f")],
        ).properties(height=300)
    )


def cohort_heatmap(retention: pd.DataFrame, max_age: int = 12) -> alt.TopLevelMixin:
    r = retention.iloc[:, : max_age + 1].copy()
    r.columns = [int(c) for c in r.columns]
    order = [f"{d:%b %y}" for d in r.index]
    r.index = order
    r.index.name = "cohort"
    long = r.reset_index().melt(id_vars="cohort", var_name="age", value_name="retention").dropna()
    long["pct"] = (long["retention"] * 100).round().astype(int).astype(str)
    base = alt.Chart(long).encode(
        x=alt.X("age:O", title="Months since first order", axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("cohort:O", sort=order, title=None),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("retention:Q", scale=alt.Scale(range=SEQ, domain=[0, 0.6], clamp=True), legend=None),
        tooltip=[alt.Tooltip("cohort:O", title="Cohort"), alt.Tooltip("age:O", title="Months later"),
                 alt.Tooltip("retention:Q", title="Retained", format=".1%")])
    text = base.mark_text(fontSize=10).encode(
        text="pct:N", color=alt.condition(alt.datum.retention > 0.33, alt.value("white"), alt.value(INK)))
    return _style((rect + text).properties(height=max(260, 17 * len(order))))


def retention_curve(retention: pd.DataFrame, max_age: int = 12) -> alt.TopLevelMixin:
    avg = retention.mean().iloc[: max_age + 1]
    d = pd.DataFrame({"age": [int(a) for a in avg.index], "retention": avg.to_numpy()})
    base = alt.Chart(d).encode(
        x=alt.X("age:Q", title="Months since first order", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("retention:Q", title="Average retention", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
    )
    return _style((base.mark_line(color=BLUE, strokeWidth=2) +
                   base.mark_circle(color=BLUE, size=60).encode(
                       tooltip=[alt.Tooltip("age:Q", title="Month"), alt.Tooltip("retention:Q", format=".1%")])
                   ).properties(height=240))


def anomaly_chart(daily: pd.DataFrame) -> alt.TopLevelMixin:
    d = daily.reset_index().rename(columns={"order_date": "date"})
    d["label"] = d["revenue"].map(inr)
    d["base_label"] = d["baseline"].map(lambda v: "" if pd.isna(v) else inr(v))
    flagged = d[d["is_anomaly"]].assign(uplift=lambda t: t["uplift_pct"].round(0))
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %y", grid=False))
    layers = [
        alt.Chart(d).mark_line(color=BLUE, strokeWidth=1).encode(
            x=x, y=alt.Y("revenue:Q", axis=_inr_axis(), scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip("label:N", title="Revenue"),
                     alt.Tooltip("base_label:N", title="28-day median")]),
        alt.Chart(d).mark_line(color=MUTED, strokeWidth=1.5, strokeDash=[4, 3]).encode(x=x, y="baseline:Q"),
    ]
    if len(flagged):
        layers.append(alt.Chart(flagged).mark_circle(color=ORANGE, size=110, stroke="white", strokeWidth=2).encode(
            x=x, y="revenue:Q",
            tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Anomaly"), alt.Tooltip("label:N", title="Revenue"),
                     alt.Tooltip("uplift:Q", title="vs baseline (%)", format="+,.0f"),
                     alt.Tooltip("robust_z:Q", title="Robust z", format=".1f")]))
    return _style(alt.layer(*layers).properties(height=340).interactive(bind_y=False))
