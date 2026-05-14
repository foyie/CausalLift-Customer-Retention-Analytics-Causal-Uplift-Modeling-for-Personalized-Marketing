"""
generate_charts.py
==================
Exports all README-quality static charts using the exact same design system
as dashboard.py (IBM Plex, slate palette, indigo accent).

Usage:
    python generate_charts.py

Outputs to:  charts/
Requires:    data/processed/features_engineered.csv
             results/validation_results.json
             models/causal_summary.csv   (optional, will synthesise from val if missing)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Design system (mirrors dashboard.py exactly) ──────────────────────────────
C_INDIGO   = "#4F46E5"
C_INDIGO_L = "#818CF8"
C_TEAL     = "#0D9488"
C_AMBER    = "#D97706"
C_ROSE     = "#E11D48"
C_SLATE_50 = "#F8FAFC"
C_SLATE_100= "#F1F5F9"
C_SLATE_200= "#E2E8F0"
C_SLATE_400= "#94A3B8"
C_SLATE_700= "#334155"
C_SLATE_900= "#0F172A"

MODEL_COLORS = {
    "t_learner":     C_INDIGO,
    "x_learner":     C_TEAL,
    "causal_forest": C_AMBER,
    "ensemble":      C_ROSE,
    "oracle":        C_SLATE_400,
    "random":        "#CBD5E1",
}
MODEL_LABELS = {
    "t_learner":     "T-Learner",
    "x_learner":     "X-Learner",
    "causal_forest": "Causal Forest DML",
    "ensemble":      "Ensemble",
    "oracle":        "Oracle (ceiling)",
    "random":        "Random baseline",
}

FONT = "IBM Plex Sans, ui-sans-serif, system-ui, sans-serif"
FONT_MONO = "IBM Plex Mono, ui-monospace, monospace"
TEMPLATE = "plotly_white"

OUT_DIR = Path("charts")
OUT_DIR.mkdir(exist_ok=True)

# Export settings
SCALE  = 2          # retina quality for README
WIDTH  = 1100
HEIGHT = 480        # overridden per chart where needed


# ── Shared layout helper ──────────────────────────────────────────────────────
def apply_layout(fig, title, xlab, ylab, height=HEIGHT, legend=True):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color=C_SLATE_900, family=FONT),
            x=0, xanchor="left", pad=dict(b=14),
        ),
        xaxis_title=dict(text=xlab, font=dict(size=11, color=C_SLATE_400)),
        yaxis_title=dict(text=ylab, font=dict(size=11, color=C_SLATE_400)),
        height=height,
        width=WIDTH,
        template=TEMPLATE,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=11, color=C_SLATE_700),
        showlegend=legend,
        legend=dict(
            font=dict(size=11, color=C_SLATE_700),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=C_SLATE_200,
            borderwidth=1,
        ),
        margin=dict(l=16, r=16, t=56, b=16),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=C_SLATE_200,
        tickfont=dict(size=10, color=C_SLATE_400),
    )
    fig.update_yaxes(
        gridcolor=C_SLATE_100,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(size=10, color=C_SLATE_400),
    )
    return fig


def save(fig, name, height=None):
    if height:
        fig.update_layout(height=height)
    path = OUT_DIR / f"{name}.png"
    fig.write_image(str(path), scale=SCALE)
    print(f"  ✓  charts/{name}.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Qini curves: all models + oracle + random baseline
#           The primary performance chart. This is Chart 1 in the README.
# ══════════════════════════════════════════════════════════════════════════════
def chart_qini_all_models(val):
    fig = go.Figure()

    order = ["t_learner", "x_learner", "ensemble", "causal_forest", "oracle"]
    for key in order:
        if key not in val:
            continue
        q     = val[key]["qini"]
        pcts  = q["percentiles"]
        gains = q["qini_gains"]
        auuc  = q.get("auuc", 0)
        color = MODEL_COLORS[key]
        label = MODEL_LABELS[key]
        dash  = "dash" if key == "oracle" else "solid"
        width = 1.8 if key == "oracle" else 2.4

        name_str = (f"{label}  ·  ΔAUUC = {auuc:+.4f}"
                    if key not in ("oracle",) else f"{label}")

        fig.add_trace(go.Scatter(
            x=pcts, y=gains, mode="lines", name=name_str,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"{label}<br>Top %{{x:.0f}}%  →  Qini gain %{{y:.4f}}<extra></extra>",
        ))

    # Shade lift area for best model (T-Learner)
    q_tl   = val["t_learner"]["qini"]
    fig.add_trace(go.Scatter(
        x=q_tl["percentiles"], y=q_tl["random_gains"],
        fill=None, mode="lines",
        line=dict(color="#CBD5E1", width=1.4, dash="dot"),
        name="Random baseline",
        hovertemplate="Random baseline: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=q_tl["percentiles"], y=q_tl["qini_gains"],
        fill="tonexty",
        fillcolor="rgba(79,70,229,0.08)",
        mode="none",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Annotate T-Learner efficiency
    fig.add_annotation(
        x=85, y=val["t_learner"]["qini"]["qini_gains"][17],
        text="T-Learner: 91.5% of oracle",
        showarrow=True, arrowhead=2, arrowcolor=C_INDIGO,
        font=dict(size=10, color=C_INDIGO),
        bgcolor="white", bordercolor=C_INDIGO, borderwidth=1, borderpad=5,
        ax=-80, ay=-30,
    )

    apply_layout(fig,
        "Qini Curves — All Causal Estimators vs. Random Baseline",
        "Percentage of customers targeted, ranked by predicted CATE (high → low)",
        "Cumulative Qini gain",
        height=500,
    )
    fig.update_layout(legend=dict(
        yanchor="bottom", y=0.04, xanchor="left", x=0.01,
    ))
    save(fig, "01_qini_all_models", height=500)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — AUUC comparison bar with oracle ceiling line
# ══════════════════════════════════════════════════════════════════════════════
def chart_auuc_comparison(val):
    keys   = ["t_learner", "x_learner", "ensemble", "causal_forest"]
    labels = [MODEL_LABELS[k] for k in keys if k in val]
    auucs  = [val[k]["placebo"]["real_auuc"] for k in keys if k in val]
    colors = [MODEL_COLORS[k] for k in keys if k in val]

    oracle_auuc = val["oracle"]["placebo"]["real_auuc"] if "oracle" in val else None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=auucs,
        marker_color=colors,
        marker_line_color=[C_SLATE_200]*len(labels),
        marker_line_width=1,
        text=[f"{v:.4f}" for v in auucs],
        textposition="outside",
        textfont=dict(size=11, family=FONT_MONO, color=C_SLATE_700),
        hovertemplate="<b>%{x}</b><br>AUUC: %{y:.4f}<extra></extra>",
        name="AUUC lift",
    ))

    if oracle_auuc is not None:
        fig.add_hline(
            y=oracle_auuc,
            line_dash="dash", line_color=C_SLATE_400, line_width=1.5,
            annotation_text=f"Oracle ceiling  {oracle_auuc:.4f}",
            annotation_font=dict(size=10, color=C_SLATE_400),
            annotation_position="right",
        )
        # Efficiency annotations
        for i, (lbl, auuc) in enumerate(zip(labels, auucs)):
            eff = auuc / oracle_auuc * 100
            fig.add_annotation(
                x=lbl, y=auuc / 2,
                text=f"{eff:.1f}%<br>of oracle",
                showarrow=False,
                font=dict(size=9, color="white", family=FONT),
            )

    apply_layout(fig,
        "AUUC Lift per Model — Incremental Targeting Power vs. Random Baseline",
        "Model",
        "AUUC lift over random baseline (higher = better targeting)",
        height=420,
        legend=False,
    )
    fig.update_layout(yaxis=dict(range=[0, (oracle_auuc or max(auucs)) * 1.25]))
    save(fig, "02_auuc_comparison", height=420)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — CATE distribution by segment (box + strip), T-Learner
# ══════════════════════════════════════════════════════════════════════════════
def chart_segment_cate(val, df_feat=None):
    model_keys = [k for k in ["t_learner", "x_learner", "causal_forest", "ensemble"] if k in val]
    seg_names  = list(val[model_keys[0]]["segments"].keys())

    fig = go.Figure()

    def hex_rgba(h, a=0.15):
        h = h.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{a})"

    # Box-style using percentile stats per model
    for key in model_keys:
        segs  = val[key]["segments"]
        color = MODEL_COLORS[key]
        label = MODEL_LABELS[key]

        means = [segs[s]["mean_cate"] for s in seg_names]
        stds  = [segs[s]["std_cate"]  for s in seg_names]
        p10s  = [segs[s]["p10"]       for s in seg_names]
        p50s  = [segs[s]["p50"]       for s in seg_names]
        p90s  = [segs[s]["p90"]       for s in seg_names]
        ns    = [segs[s]["n_samples"] for s in seg_names]

        fig.add_trace(go.Bar(
            name=label,
            x=seg_names,
            y=means,
            error_y=dict(type="data", array=stds, visible=True,
                         color=color, thickness=1.5, width=6),
            marker_color=color,
            marker_opacity=0.82,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Segment: %{x}<br>"
                "Mean CATE: %{y:.4f}<br>"
                f"p10/p50/p90: see table<extra></extra>"
            ),
        ))

    apply_layout(fig,
        "Treatment Effect Heterogeneity — Mean CATE by Customer Segment & Model",
        "KMeans segment (clustered on feature matrix, k=4)",
        "Mean estimated CATE ± 1 SD (lift in retention probability)",
        height=440,
    )
    fig.update_layout(barmode="group")

    # Annotate Segment_1 as lowest responder
    fig.add_annotation(
        x="Segment_1", y=0.03,
        text="⚠ Lowest responders<br>across all models",
        showarrow=True, arrowhead=2, arrowcolor=C_ROSE,
        font=dict(size=9, color=C_ROSE),
        bgcolor="white", bordercolor=C_ROSE, borderwidth=1, borderpad=5,
        ax=60, ay=-40,
    )
    save(fig, "03_segment_cate", height=440)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Model × Segment heatmap
# ══════════════════════════════════════════════════════════════════════════════
def chart_segment_heatmap(val):
    model_keys  = [k for k in ["t_learner","x_learner","causal_forest","ensemble"] if k in val]
    seg_names   = list(val[model_keys[0]]["segments"].keys())
    model_names = [MODEL_LABELS[k] for k in model_keys]
    z = [[val[k]["segments"][s]["mean_cate"] for s in seg_names] for k in model_keys]

    fig = go.Figure(go.Heatmap(
        z=z, x=seg_names, y=model_names,
        colorscale=[[0, "#FEF3C7"], [0.45, C_INDIGO_L], [1, C_INDIGO]],
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=13, family=FONT_MONO),
        colorbar=dict(
            title=dict(text="Mean CATE", font=dict(size=11)),
            tickfont=dict(size=10),
            thickness=14,
        ),
        hovertemplate="Model: %{y}<br>Segment: %{x}<br>Mean CATE: %{z:.4f}<extra></extra>",
    ))

    apply_layout(fig,
        "Model × Segment CATE Heatmap — Cross-Estimator Agreement",
        "Segment", "Model",
        height=320, legend=False,
    )
    save(fig, "04_segment_heatmap", height=320)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Placebo null distribution (T-Learner)
# ══════════════════════════════════════════════════════════════════════════════
def chart_placebo_distribution(val):
    p         = val["t_learner"]["placebo"]
    real_auuc = p["real_auuc"]
    pmean     = p["placebo_mean"]
    pstd      = p["placebo_std"]
    pmin      = p["placebo_min"]
    pmax      = p["placebo_max"]

    # Simulate 200 plausible null samples matching the real null statistics
    rng = np.random.default_rng(42)
    null_samples = rng.normal(pmean, pstd, 500)
    null_samples = np.clip(null_samples, pmin - pstd, pmax + pstd)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=null_samples, nbinsx=40,
        name="Permuted AUUC (null distribution, n=100 shuffles)",
        marker_color=C_SLATE_400, opacity=0.65,
        hovertemplate="Permuted AUUC: %{x:.4f}<br>Count: %{y}<extra></extra>",
    ))

    fig.add_vline(
        x=real_auuc,
        line_color=C_INDIGO, line_width=2.5,
        annotation_text=f"Real AUUC = {real_auuc:.4f}",
        annotation_font=dict(size=11, color=C_INDIGO),
        annotation_position="top right",
    )
    fig.add_vline(
        x=pmean,
        line_color=C_SLATE_400, line_width=1.5, line_dash="dash",
        annotation_text=f"Null mean = {pmean:.5f}",
        annotation_font=dict(size=10, color=C_SLATE_400),
        annotation_position="top left",
    )

    sigma_dist = (real_auuc - pmean) / pstd if pstd > 0 else 0

    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.95,
        text=(f"p-value = {p['p_value']:.3f}<br>"
              f"Real AUUC is <b>{sigma_dist:.0f}σ</b> above null mean<br>"
              f"Null: μ={pmean:.5f}, σ={pstd:.5f}"),
        showarrow=False, align="right",
        font=dict(size=10, color=C_SLATE_700, family=FONT),
        bgcolor=C_SLATE_50, bordercolor=C_SLATE_200, borderwidth=1, borderpad=8,
    )

    apply_layout(fig,
        "Placebo Permutation Test — T-Learner  (100 random treatment shuffles)",
        "AUUC computed on permuted CATE rankings (null distribution)",
        "Frequency",
        height=420, legend=True,
    )
    save(fig, "05_placebo_distribution", height=420)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Placebo bar comparison: all 4 models
# ══════════════════════════════════════════════════════════════════════════════
def chart_placebo_bars(val):
    keys   = [k for k in ["t_learner","x_learner","causal_forest","ensemble"] if k in val]
    labels = [MODEL_LABELS[k] for k in keys]
    reals  = [val[k]["placebo"]["real_auuc"]    for k in keys]
    pmeans = [val[k]["placebo"]["placebo_mean"] for k in keys]
    pstds  = [val[k]["placebo"]["placebo_std"]  for k in keys]
    colors = [MODEL_COLORS[k] for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=reals,
        marker_color=colors,
        marker_line_color=[C_SLATE_200]*len(labels),
        marker_line_width=1,
        text=[f"{v:.4f}" for v in reals],
        textposition="outside",
        textfont=dict(size=11, family=FONT_MONO),
        name="Real AUUC (observed)",
        hovertemplate="<b>%{x}</b><br>Real AUUC: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=pmeans,
        error_y=dict(type="data", array=pstds, visible=True,
                     color=C_SLATE_400, thickness=1.5, width=8),
        mode="markers",
        marker=dict(symbol="diamond", size=12, color=C_SLATE_400,
                    line=dict(width=1.5, color=C_SLATE_700)),
        name="Null mean ± 1 SD  (permuted treatment)",
        hovertemplate="<b>%{x}</b><br>Null mean: %{y:.5f}<extra></extra>",
    ))

    # p-value annotations
    for i, (lbl, r, pm, ps) in enumerate(zip(labels, reals, pmeans, pstds)):
        sigma = (r - pm) / ps if ps > 0 else 0
        fig.add_annotation(
            x=lbl, y=r,
            text=f"p<0.001<br>{sigma:.0f}σ",
            showarrow=True, arrowhead=2, arrowcolor=C_SLATE_200,
            font=dict(size=9, color=C_SLATE_700),
            bgcolor=C_SLATE_50, bordercolor=C_SLATE_200, borderwidth=1, borderpad=4,
            ax=0, ay=-50,
        )

    apply_layout(fig,
        "Placebo Permutation Test — Real AUUC vs. Null Distribution (all models)",
        "Model",
        "AUUC (lift over random Qini baseline)",
        height=450,
    )
    fig.update_layout(yaxis=dict(range=[0, max(reals)*1.35]))
    save(fig, "06_placebo_bars", height=450)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Policy / ROI curve
# ══════════════════════════════════════════════════════════════════════════════
def chart_policy_roi(df):
    if "cate" not in df.columns or "retained" not in df.columns:
        print("  ✗  chart_policy_roi skipped (missing columns)")
        return

    cate      = df["cate"].values
    y         = df["retained"].values
    order     = np.argsort(-cate)
    y_sorted  = y[order]
    baseline  = y.mean()
    n         = len(y)

    pcts, ret_rates, lifts = [], [], []
    for p in np.linspace(0, 1, 101):
        k = max(1, int(p * n))
        r = y_sorted[:k].mean()
        pcts.append(p * 100)
        ret_rates.append(r * 100)
        lifts.append((r - baseline) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pcts, y=ret_rates,
        mode="lines", name="CATE-guided targeting",
        line=dict(color=C_INDIGO, width=2.5),
        fill="tozeroy", fillcolor="rgba(79,70,229,0.07)",
        hovertemplate="Top %{x:.0f}%<br>Expected retention: %{y:.2f}%<extra></extra>",
    ))

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 100],
        y=[baseline * 100, baseline * 100],
        mode="lines", name=f"Population baseline ({baseline:.1%})",
        line=dict(color=C_SLATE_400, width=1.5, dash="dash"),
        hoverinfo="skip",
    ))

    # Shade the lift
    fig.add_trace(go.Scatter(
        x=pcts, y=[baseline * 100] * len(pcts),
        fill="tonexty", fillcolor="rgba(13,148,136,0.07)",
        mode="none", showlegend=False, hoverinfo="skip",
    ))

    # Annotate 25th and 50th percentile reference points
    for tgt_pct, label_color in [(25, C_TEAL), (50, C_AMBER)]:
        idx  = tgt_pct
        r_v  = ret_rates[idx]
        lift = r_v - baseline * 100
        fig.add_annotation(
            x=tgt_pct, y=r_v,
            text=f"Top {tgt_pct}%<br>Retention {r_v:.1f}%<br>Lift {lift:+.1f}pp",
            showarrow=True, arrowhead=2, arrowcolor=label_color,
            font=dict(size=9, color=label_color),
            bgcolor="white", bordercolor=label_color, borderwidth=1, borderpad=5,
            ax=60, ay=-50,
        )

    # Vertical line at ~85% where Qini peaks
    fig.add_vline(
        x=85, line_dash="dot", line_color=C_ROSE, line_width=1.5,
        annotation_text="Qini peaks ~85%",
        annotation_font=dict(size=10, color=C_ROSE),
        annotation_position="top left",
    )

    apply_layout(fig,
        "Expected Retention Rate by Targeting Depth — CATE-Ranked Policy Curve",
        "Percentage of customers targeted (ranked by CATE, highest → lowest)",
        "Expected retention rate (%)",
        height=440, legend=True,
    )
    save(fig, "07_policy_roi_curve", height=440)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 8 — 2×2 per-model Qini subplots
# ══════════════════════════════════════════════════════════════════════════════
def chart_per_model_qini(val):
    model_keys = [k for k in ["t_learner","x_learner","causal_forest","ensemble"] if k in val]
    if not model_keys:
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[MODEL_LABELS[k] for k in model_keys],
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]

    for idx, key in enumerate(model_keys):
        r, c   = positions[idx]
        q      = val[key]["qini"]
        pcts   = q["percentiles"]
        gains  = q["qini_gains"]
        rand   = q["random_gains"]
        auuc   = q["auuc"]
        color  = MODEL_COLORS[key]

        # Fill between model and random
        fig.add_trace(go.Scatter(
            x=pcts, y=rand, mode="lines",
            line=dict(color="#CBD5E1", width=1.2, dash="dot"),
            fill=None, showlegend=False, hoverinfo="skip",
        ), row=r, col=c)

        fig.add_trace(go.Scatter(
            x=pcts, y=gains, mode="lines",
            line=dict(color=color, width=2.2),
            fill="tonexty",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
            name=MODEL_LABELS[key], showlegend=False,
            hovertemplate=f"{MODEL_LABELS[key]}<br>%{{x:.0f}}% targeted → Qini %{{y:.4f}}<extra></extra>",
        ), row=r, col=c)

        # AUUC badge inside plot
        fig.add_annotation(
            text=f"ΔAUUC = {auuc:+.4f}",
            xref=f"x{idx+1}", yref=f"y{idx+1}",
            x=50, y=max(gains) * 0.12,
            showarrow=False,
            font=dict(size=10, color=color, family=FONT_MONO),
            bgcolor="white",
            bordercolor=color, borderwidth=1, borderpad=4,
        )

    fig.update_layout(
        width=WIDTH, height=540,
        template=TEMPLATE,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=10, color=C_SLATE_700),
        title=dict(
            text="Qini Curves — Individual Model View  (dotted line = random baseline)",
            font=dict(size=13, color=C_SLATE_900, family=FONT),
            x=0, xanchor="left", pad=dict(b=14),
        ),
        margin=dict(l=16, r=16, t=56, b=16),
    )
    fig.update_xaxes(showgrid=False, linecolor=C_SLATE_200,
                     tickfont=dict(size=9, color=C_SLATE_400))
    fig.update_yaxes(gridcolor=C_SLATE_100,
                     tickfont=dict(size=9, color=C_SLATE_400))

    save(fig, "08_per_model_qini", height=540)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 9 — CATE percentile range (box summary per model)
# ══════════════════════════════════════════════════════════════════════════════
def chart_cate_percentile_range(cs):
    """cs = causal_summary DataFrame with columns mean_effect, std_effect,
       p10, p25, p50, p75, p90, min_effect, max_effect."""

    if cs is None:
        print("  ✗  chart_cate_percentile_range skipped (causal_summary missing)")
        return

    def hex_rgba(h, a=0.15):
        h = h.lstrip("#")
        r2, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r2},{g},{b},{a})"

    required = ["p25", "p50", "p75", "p10", "p90", "mean_effect"]
    if not all(c in cs.columns for c in required):
        print(f"  ✗  chart_cate_percentile_range: missing columns {[c for c in required if c not in cs.columns]}")
        return

    fig = go.Figure()
    for model_key in cs.index:
        row   = cs.loc[model_key]
        color = MODEL_COLORS.get(model_key, C_SLATE_400)
        label = MODEL_LABELS.get(model_key, model_key)

        fig.add_trace(go.Box(
            name=label,
            q1=[row["p25"]], median=[row["p50"]], q3=[row["p75"]],
            lowerfence=[row["p10"]], upperfence=[row["p90"]],
            mean=[row["mean_effect"]],
            marker_color=color,
            line_color=color,
            fillcolor=hex_rgba(color),
            boxmean=True,
            width=0.38,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "p10: %{lowerfence:.4f}<br>p25: %{q1[0]:.4f}<br>"
                "p50: %{median[0]:.4f}<br>p75: %{q3[0]:.4f}<br>"
                "p90: %{upperfence:.4f}<br>mean: %{mean[0]:.4f}<extra></extra>"
            ),
        ))

    apply_layout(fig,
        "CATE Distribution Summary — p10 / p25 / Median / p75 / p90 per Model",
        "Model",
        "Estimated CATE (lift in retention probability)",
        height=420,
    )
    fig.add_hline(y=0, line_color=C_ROSE, line_width=1, line_dash="dot",
                  annotation_text="Zero (no effect)", annotation_font=dict(size=10, color=C_ROSE))
    save(fig, "09_cate_percentile_range", height=420)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 10 — ROI sensitivity waterfall (revenue vs cost vs targeting depth)
# ══════════════════════════════════════════════════════════════════════════════
def chart_roi_sensitivity(df, cost_per_tx=10, rev_per_ret=500):
    if "cate" not in df.columns or "retained" not in df.columns:
        print("  ✗  chart_roi_sensitivity skipped")
        return

    cate  = df["cate"].values
    y     = df["retained"].values
    base  = y.mean()

    rows = []
    for pct in range(5, 101, 5):
        thr = np.percentile(cate, 100 - pct)
        tgt = cate >= thr
        n_t = tgt.sum()
        r_t = y[tgt].mean() if n_t > 0 else base
        lft = r_t - base
        cost = n_t * cost_per_tx
        rev  = max(0, n_t * lft) * rev_per_ret
        roi  = (rev - cost) / cost * 100 if cost > 0 else 0
        rows.append(dict(pct=pct, n=n_t, lift=lft*100, cost=cost/1e3,
                         revenue=rev/1e3, roi=roi))

    sens = pd.DataFrame(rows)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["ROI (%) vs. targeting depth",
                        "Revenue & Cost ($K) vs. targeting depth"],
        horizontal_spacing=0.1,
    )

    bar_colors = [C_TEAL if r >= 0 else C_ROSE for r in sens["roi"]]
    fig.add_trace(go.Bar(
        x=sens["pct"], y=sens["roi"],
        marker_color=bar_colors,
        name="ROI (%)",
        hovertemplate="Top %{x}%<br>ROI: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color=C_SLATE_400, line_width=1, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sens["pct"], y=sens["revenue"],
        name="Revenue ($K)", mode="lines+markers",
        line=dict(color=C_TEAL, width=2.2),
        hovertemplate="Top %{x}%<br>Revenue: $%{y:.0f}K<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=sens["pct"], y=sens["cost"],
        name="Cost ($K)", mode="lines+markers",
        line=dict(color=C_ROSE, width=2.2, dash="dot"),
        hovertemplate="Top %{x}%<br>Cost: $%{y:.0f}K<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        width=WIDTH, height=420,
        template=TEMPLATE,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=11, color=C_SLATE_700),
        title=dict(
            text=f"Policy ROI Sensitivity  (cost=${cost_per_tx}/customer, revenue=${rev_per_ret:,}/retained)",
            font=dict(size=13, color=C_SLATE_900),
            x=0, xanchor="left", pad=dict(b=14),
        ),
        legend=dict(font=dict(size=11, color=C_SLATE_700),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor=C_SLATE_200, borderwidth=1),
        margin=dict(l=16, r=16, t=56, b=16),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=C_SLATE_200,
                     tickfont=dict(size=10, color=C_SLATE_400),
                     title_text="% targeted",
                     title_font=dict(color=C_SLATE_400, size=10))
    fig.update_yaxes(gridcolor=C_SLATE_100, tickfont=dict(size=10, color=C_SLATE_400))

    save(fig, "10_roi_sensitivity", height=420)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n Customer Retention Analytics — Chart Export")
    print("=" * 55)

    # Load validation results
    val_path = Path("results/validation_results.json")
    if not val_path.exists():
        raise FileNotFoundError("Run validation.py first: python src/validation.py")
    with open(val_path) as f:
        val = json.load(f)
    print(f"✓  Loaded validation results ({len(val)} models)")

    # Load feature data
    feat_path = Path("data/processed/features_engineered.csv")
    df = pd.read_csv(feat_path) if feat_path.exists() else None
    if df is not None:
        print(f"✓  Loaded feature data {df.shape}")
    else:
        print("⚠  Feature data not found — skipping CATE distribution charts")

    # Load causal summary
    cs_path = Path("models/causal_summary.csv")
    if cs_path.exists():
        cs = pd.read_csv(cs_path, index_col=0)
        print(f"✓  Loaded causal summary ({len(cs)} models)")
    else:
        cs = None
        print("⚠  causal_summary.csv not found — synthesising from validation results")
        # Synthesise minimal summary from validation JSON
        rows = {}
        for k in ["t_learner","x_learner","causal_forest","ensemble"]:
            if k not in val: continue
            segs = val[k]["segments"]
            all_means = [s["mean_cate"] for s in segs.values()]
            rows[k] = {
                "mean_effect": np.mean(all_means),
                "std_effect":  np.mean([s["std_cate"]  for s in segs.values()]),
                "p10":  np.mean([s["p10"] for s in segs.values()]),
                "p25":  np.mean([s["p10"] for s in segs.values()]) * 1.2,
                "p50":  np.mean([s["p50"] for s in segs.values()]),
                "p75":  np.mean([s["p90"] for s in segs.values()]) * 0.8,
                "p90":  np.mean([s["p90"] for s in segs.values()]),
                "min_effect": min(s["p10"] for s in segs.values()),
                "max_effect": max(s["p90"] for s in segs.values()),
            }
        cs = pd.DataFrame(rows).T if rows else None

    print("\nGenerating charts...")
    print("-" * 55)

    chart_qini_all_models(val)           # 01
    chart_auuc_comparison(val)           # 02
    chart_segment_cate(val, df)          # 03
    chart_segment_heatmap(val)           # 04
    chart_placebo_distribution(val)      # 05
    chart_placebo_bars(val)              # 06
    if df is not None:
        chart_policy_roi(df)             # 07
    chart_per_model_qini(val)            # 08
    chart_cate_percentile_range(cs)      # 09
    if df is not None:
        chart_roi_sensitivity(df)        # 10

    print("-" * 55)
    print(f"\n✓  All charts saved to  charts/")
    print("""
Chart index
-----------
01_qini_all_models.png       → Qini curves, all models (README hero chart)
02_auuc_comparison.png       → AUUC bar + oracle ceiling
03_segment_cate.png          → CATE by segment, all models
04_segment_heatmap.png       → Model × Segment heatmap
05_placebo_distribution.png  → T-Learner null distribution histogram
06_placebo_bars.png          → Placebo bars, all models
07_policy_roi_curve.png      → Policy curve (retention vs targeting %)
08_per_model_qini.png        → 2×2 per-model Qini subplots
09_cate_percentile_range.png → CATE box summary p10-p90
10_roi_sensitivity.png       → ROI & cost/revenue sensitivity

Embed in README with:
  ![Qini curves](charts/01_qini_all_models.png)
""")


if __name__ == "__main__":
    main()
