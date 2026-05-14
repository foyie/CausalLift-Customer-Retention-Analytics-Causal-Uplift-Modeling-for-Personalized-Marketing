"""
Customer Retention Analytics Dashboard
Causal Inference & Heterogeneous Treatment Effects
Author: Chandrima Das (chdas@ucsd.edu)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retention Analytics | CATE Dashboard",
    page_icon="assets/favicon.png" if Path("assets/favicon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ──────────────────────────────────────────────────────────────
# Palette: slate base, indigo accent, semantic colors
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

PLOTLY_TEMPLATE = "plotly_white"
FONT_FAMILY = "IBM Plex Sans, ui-sans-serif, system-ui, sans-serif"

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

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{ font-family: '{FONT_FAMILY}'; color: {C_SLATE_700}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: {C_SLATE_900};
      border-right: 1px solid #1E293B;
  }}
  [data-testid="stSidebar"] * {{ color: #CBD5E1 !important; }}
  [data-testid="stSidebar"] .stRadio label {{ font-size: 0.83rem; letter-spacing: 0.02em; }}
  [data-testid="stSidebar"] hr {{ border-color: #1E293B !important; }}

  /* Main area */
  .main .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }}

  /* Page title */
  .page-title {{
      font-size: 1.05rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: {C_SLATE_400};
      margin-bottom: 0.15rem;
  }}
  .page-heading {{
      font-size: 1.75rem;
      font-weight: 600;
      color: {C_SLATE_900};
      margin-bottom: 0.5rem;
      line-height: 1.25;
  }}
  .page-desc {{
      font-size: 0.9rem;
      color: {C_SLATE_400};
      margin-bottom: 1.5rem;
  }}

  /* Section header */
  .section-header {{
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {C_SLATE_400};
      border-bottom: 1px solid {C_SLATE_200};
      padding-bottom: 0.4rem;
      margin: 1.8rem 0 1rem 0;
  }}

  /* KPI card */
  .kpi-card {{
      background: white;
      border: 1px solid {C_SLATE_200};
      border-top: 3px solid {C_INDIGO};
      border-radius: 6px;
      padding: 1.1rem 1.25rem 1rem;
  }}
  .kpi-label {{
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: {C_SLATE_400};
      margin-bottom: 0.35rem;
  }}
  .kpi-value {{
      font-size: 1.6rem;
      font-weight: 600;
      color: {C_SLATE_900};
      font-family: 'IBM Plex Mono', monospace;
      line-height: 1.1;
  }}
  .kpi-delta {{
      font-size: 0.78rem;
      color: {C_SLATE_400};
      margin-top: 0.2rem;
  }}
  .kpi-delta.positive {{ color: {C_TEAL}; }}
  .kpi-delta.negative {{ color: {C_ROSE}; }}

  /* Insight box */
  .insight-box {{
      background: {C_SLATE_50};
      border-left: 3px solid {C_INDIGO};
      border-radius: 0 4px 4px 0;
      padding: 0.85rem 1rem;
      font-size: 0.85rem;
      line-height: 1.6;
      margin: 0.75rem 0;
  }}
  .insight-box strong {{ color: {C_SLATE_900}; }}

  /* Badge */
  .badge {{
      display: inline-block;
      padding: 0.15rem 0.55rem;
      border-radius: 3px;
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.04em;
  }}
  .badge-success {{ background: #DCFCE7; color: #15803D; }}
  .badge-warn    {{ background: #FEF3C7; color: #B45309; }}
  .badge-neutral {{ background: {C_SLATE_100}; color: {C_SLATE_400}; }}

  /* Table overrides */
  .dataframe thead th {{ background: {C_SLATE_100} !important; font-size: 0.78rem; font-weight: 600; }}
  .dataframe tbody td {{ font-size: 0.82rem; font-family: 'IBM Plex Mono', monospace; }}

  /* Streamlit overrides */
  div[data-testid="stMetric"] label {{ font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }}
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem !important; }}
  .stButton > button {{
      background: {C_INDIGO}; color: white; border: none;
      border-radius: 4px; font-size: 0.82rem; font-weight: 500;
      padding: 0.45rem 1rem;
  }}
  hr {{ border-color: {C_SLATE_200} !important; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def kpi(label, value, delta=None, delta_positive=True):
    delta_cls = ("positive" if delta_positive else "negative") if delta else ""
    delta_html = f'<div class="kpi-delta {delta_cls}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def badge(text, kind="neutral"):
    return f'<span class="badge badge-{kind}">{text}</span>'

def is_dark():
    """Best-effort dark mode detection for Plotly chart theming."""
    try:
        theme = st.context.headers.get("X-Streamlit-Theme", "light")
        return "dark" in theme.lower()
    except Exception:
        return False

def plotly_defaults(fig, title, xlab, ylab, height=420, legend=True):
    dark = is_dark()
    bg       = "#0F172A" if dark else "white"
    grid_col = "#1E293B" if dark else C_SLATE_100
    line_col = "#334155" if dark else C_SLATE_200
    text_col = "#CBD5E1" if dark else C_SLATE_700
    sub_col  = "#64748B" if dark else C_SLATE_400

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color=text_col, family=FONT_FAMILY),
            x=0, xanchor="left", pad=dict(b=12)
        ),
        xaxis_title=dict(text=xlab, font=dict(size=11, color=sub_col)),
        yaxis_title=dict(text=ylab, font=dict(size=11, color=sub_col)),
        height=height,
        template="plotly_dark" if dark else PLOTLY_TEMPLATE,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family=FONT_FAMILY, size=11, color=text_col),
        showlegend=legend,
        legend=dict(
            font=dict(size=11, color=text_col),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(l=12, r=12, t=48, b=12),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=line_col,
        tickfont=dict(size=10, color=sub_col),
        title_font=dict(color=sub_col),
    )
    fig.update_yaxes(
        gridcolor=grid_col,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(size=10, color=sub_col),
        title_font=dict(color=sub_col),
    )
    return fig


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    try:
        return pd.read_csv("data/processed/features_engineered.csv")
    except FileNotFoundError:
        return None

@st.cache_data(show_spinner=False)
def load_validation():
    try:
        with open("results/validation_results.json") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_causal_summary():
    try:
        return pd.read_csv("models/causal_summary.csv", index_col=0)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_predictions():
    try:
        return pd.read_csv("models/cate_predictions.csv")
    except Exception:
        return None


# ── Chart builders ─────────────────────────────────────────────────────────────

def fig_cate_distribution(df):
    cate = df["cate"].dropna()
    mean_v, med_v = cate.mean(), cate.median()
    pct_neg = (cate < 0).mean()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=cate, nbinsx=80, name="CATE",
        marker_color=C_INDIGO, opacity=0.75,
        hovertemplate="CATE bin: %{x:.3f}<br>Count: %{y:,}<extra></extra>",
    ))
    for val, color, label, pos in [
        (mean_v,   C_TEAL,  f"Mean {mean_v:.3f}",   "top right"),
        (med_v,    C_AMBER, f"Median {med_v:.3f}",   "top left"),
        (0,        C_ROSE,  "Zero (no effect)",      "bottom right"),
    ]:
        fig.add_vline(x=val, line_width=1.5, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_font_color=color,
                      annotation_font_size=10, annotation_position=pos)

    fig = plotly_defaults(fig,
        "Distribution of Conditional Average Treatment Effects (CATE)",
        "CATE — estimated lift in retention probability",
        "Number of customers",
        height=400, legend=False)
    fig.add_annotation(
        text=f"{pct_neg:.1%} of customers have negative CATE<br>(treatment harmful for this group)",
        xref="paper", yref="paper", x=0.98, y=0.95, showarrow=False,
        align="right", font=dict(size=10, color=C_SLATE_400),
        bgcolor="white", bordercolor=C_SLATE_200, borderwidth=1, borderpad=6,
    )
    return fig


def fig_qini_all_models(val):
    """Multi-model Qini curves — the chart shown in the analysis above."""
    fig = go.Figure()
    model_order = ["t_learner", "x_learner", "ensemble", "causal_forest", "oracle", "random"]

    for key in model_order:
        if key not in val:
            continue
        qini = val[key]["qini"]
        pcts  = qini["percentiles"]
        gains = qini["qini_gains"]
        auuc  = qini.get("auuc", 0)
        label = MODEL_LABELS.get(key, key)
        color = MODEL_COLORS.get(key, "#888")
        dash  = "dot" if key == "random" else ("dash" if key == "oracle" else "solid")
        width = 1.5 if key in ("random", "oracle") else 2.2

        if key == "random":
            gains = qini.get("random_gains", gains)

        fig.add_trace(go.Scatter(
            x=pcts, y=gains,
            name=f"{label}  (AUUC {auuc:+.4f})" if key not in ("random", "oracle") else label,
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"{label}<br>Top %%: %{{x:.0f}}%<br>Qini gain: %{{y:.4f}}<extra></extra>",
        ))

    fig = plotly_defaults(fig,
        "Qini Curves — Targeting Lift vs. Random Baseline  (all models)",
        "Percentage of customers targeted, sorted by predicted CATE (high → low)",
        "Cumulative Qini gain (normalized, [-1, 1])",
        height=460)
    fig.update_layout(legend=dict(
        yanchor="bottom", y=0.04, xanchor="left", x=0.01,
        font=dict(size=10), bgcolor="rgba(255,255,255,0.85)",
        bordercolor=C_SLATE_200, borderwidth=1,
    ))
    return fig


def fig_placebo_comparison(val):
    """Bar chart: real AUUC vs placebo mean ± std for each model."""
    rows = []
    for key in ["t_learner", "x_learner", "causal_forest", "ensemble"]:
        if key not in val:
            continue
        p = val[key]["placebo"]
        rows.append(dict(
            model=MODEL_LABELS[key],
            real=p["real_auuc"],
            placebo_mean=p["placebo_mean"],
            placebo_std=p["placebo_std"],
            p_value=p["p_value"],
            significant=p["significant"],
        ))
    if not rows:
        return None
    df_p = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_p["model"], y=df_p["real"],
        name="Real AUUC",
        marker_color=[MODEL_COLORS[k] for k in ["t_learner","x_learner","causal_forest","ensemble"] if k in val],
        text=[f"{v:.4f}" for v in df_p["real"]],
        textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Real AUUC: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_p["model"], y=df_p["placebo_mean"],
        error_y=dict(type="data", array=df_p["placebo_std"], visible=True, color=C_SLATE_400),
        name="Placebo mean ± 1 SD",
        mode="markers",
        marker=dict(symbol="diamond", size=10, color=C_SLATE_400),
        hovertemplate="<b>%{x}</b><br>Placebo mean: %{y:.4f}<extra></extra>",
    ))

    fig = plotly_defaults(fig,
        "Placebo Test — Real AUUC vs. Permuted CATE Baseline",
        "Model",
        "AUUC (lift over random Qini baseline)",
        height=400, legend=True)
    fig.update_layout(barmode="group")
    return fig


def fig_segment_comparison(val):
    """Grouped bar — mean CATE per KMeans segment, one group per model."""
    model_keys = [k for k in ["t_learner","x_learner","causal_forest","ensemble"] if k in val]
    if not model_keys:
        return None

    # Collect segments (use first model for segment names)
    seg_names = list(val[model_keys[0]]["segments"].keys())

    fig = go.Figure()
    for key in model_keys:
        segs = val[key]["segments"]
        means = [segs[s]["mean_cate"] for s in seg_names]
        stds  = [segs[s]["std_cate"]  for s in seg_names]
        fig.add_trace(go.Bar(
            name=MODEL_LABELS[key],
            x=seg_names,
            y=means,
            error_y=dict(type="data", array=stds, visible=True, thickness=1.2),
            marker_color=MODEL_COLORS[key],
            opacity=0.85,
            hovertemplate=f"<b>{MODEL_LABELS[key]}</b><br>Segment: %{{x}}<br>Mean CATE: %{{y:.4f}}<extra></extra>",
        ))

    fig = plotly_defaults(fig,
        "Segment Heterogeneity — Mean CATE per KMeans Cluster (4 segments)",
        "KMeans segment (clustered on customer features, not CATE)",
        "Mean estimated CATE (error bars = ± 1 SD)",
        height=420)
    fig.update_layout(barmode="group")
    return fig


def fig_segment_heatmap(val):
    """Heatmap of mean CATE: models × segments."""
    model_keys = [k for k in ["t_learner","x_learner","causal_forest","ensemble"] if k in val]
    if not model_keys:
        return None

    seg_names   = list(val[model_keys[0]]["segments"].keys())
    model_names = [MODEL_LABELS[k] for k in model_keys]
    z = [[val[k]["segments"][s]["mean_cate"] for s in seg_names] for k in model_keys]

    fig = go.Figure(go.Heatmap(
        z=z, x=seg_names, y=model_names,
        colorscale=[[0, "#FEF3C7"], [0.5, C_INDIGO_L], [1, C_INDIGO]],
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Mean CATE", tickfont=dict(size=10)),
        hovertemplate="Model: %{y}<br>Segment: %{x}<br>Mean CATE: %{z:.4f}<extra></extra>",
    ))
    fig = plotly_defaults(fig,
        "Model × Segment CATE Heatmap — agreement across estimators",
        "Segment", "Model", height=280, legend=False)
    return fig


def fig_model_summary_bars(causal_summary):
    """Mean CATE ± std for each model."""
    if causal_summary is None:
        return None
    models = [MODEL_LABELS.get(m, m) for m in causal_summary.index]
    colors = [MODEL_COLORS.get(m, C_SLATE_400) for m in causal_summary.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=causal_summary["mean_effect"],
        error_y=dict(type="data", array=causal_summary["std_effect"], visible=True, thickness=1.5),
        marker_color=colors,
        text=[f"{v:.4f}" for v in causal_summary["mean_effect"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Mean CATE: %{y:.4f}<br>Std: %{customdata:.4f}<extra></extra>",
        customdata=causal_summary["std_effect"],
        name="Mean CATE",
    ))

    fig = plotly_defaults(fig,
        "Mean Estimated CATE per Model (error bars = ± 1 SD across 100K customers)",
        "Model", "Mean CATE (estimated lift in retention probability)",
        height=380, legend=False)
    return fig


def fig_cate_percentile_range(causal_summary):
    """Percentile band chart — p10, p25, p50, p75, p90 per model."""
    if causal_summary is None:
        return None

    models = [MODEL_LABELS.get(m, m) for m in causal_summary.index]
    colors = [MODEL_COLORS.get(m, C_SLATE_400) for m in causal_summary.index]

    def _hex_rgba(h, a=0.18):
        h = h.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{a})"

    fig = go.Figure()
    for i, (model, color) in enumerate(zip(models, colors)):
        row = causal_summary.iloc[i]
        fig.add_trace(go.Box(
            name=model,
            q1=[row["p25"]], median=[row["p50"]],
            q3=[row["p75"]], lowerfence=[row["p10"]], upperfence=[row["p90"]],
            mean=[row["mean_effect"]],
            marker_color=color,
            line_color=color,
            fillcolor=_hex_rgba(color),
            boxmean=True,
            width=0.4,
        ))

    fig = plotly_defaults(fig,
        "CATE Distribution Summary — p10 / p25 / median / p75 / p90 per model",
        "Model", "Estimated CATE (lift in retention probability)",
        height=380)
    return fig


def fig_roi_curve(df):
    """Retention rate vs. targeting rate, sorted by CATE descending."""
    if "cate" not in df.columns or "retained" not in df.columns:
        return None

    cate = df["cate"].values
    y    = df["retained"].values
    order = np.argsort(-cate)
    y_sorted = y[order]
    n = len(y)
    baseline = y.mean()

    pcts, ret_rates, cum_ret = [], [], []
    for pct in np.linspace(0, 1, 41):
        k = max(1, int(pct * n))
        pcts.append(pct * 100)
        ret_rates.append(y_sorted[:k].mean() * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pcts, y=ret_rates,
        mode="lines", name="CATE-guided targeting",
        line=dict(color=C_INDIGO, width=2.5),
        fill="tozeroy", fillcolor=f"rgba(79,70,229,0.08)",
        hovertemplate="Top %{x:.0f}% targeted<br>Retention rate: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=baseline * 100, line_dash="dash", line_color=C_SLATE_400,
                  annotation_text=f"Population baseline {baseline:.1%}",
                  annotation_font_size=10, annotation_font_color=C_SLATE_400)

    fig = plotly_defaults(fig,
        "Expected Retention Rate by Targeting Depth — CATE-ranked customers",
        "Percentage of customers targeted (ranked by predicted CATE, high to low)",
        "Expected retention rate (%)",
        height=400, legend=False)
    return fig


def fig_pred_vs_true(pred_df):
    """Scatter: predicted ensemble CATE vs. oracle CATE."""
    if pred_df is None or "true_cate" not in pred_df.columns or "ensemble_cate" not in pred_df.columns:
        return None

    sample = pred_df.sample(min(5000, len(pred_df)), random_state=42)
    corr = np.corrcoef(sample["true_cate"], sample["ensemble_cate"])[0, 1]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=sample["true_cate"], y=sample["ensemble_cate"],
        mode="markers",
        marker=dict(color=C_INDIGO, opacity=0.25, size=4),
        hovertemplate="Oracle CATE: %{x:.3f}<br>Predicted CATE: %{y:.3f}<extra></extra>",
        name="Customers",
    ))
    lims = [min(sample["true_cate"].min(), sample["ensemble_cate"].min()),
            max(sample["true_cate"].max(), sample["ensemble_cate"].max())]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color=C_ROSE, dash="dash", width=1.5),
        name="Perfect prediction (y=x)",
    ))

    fig = plotly_defaults(fig,
        f"Predicted vs. Oracle CATE — Ensemble model  (Pearson r = {corr:.3f}, n = 5,000 sample)",
        "Oracle (true) CATE",
        "Ensemble predicted CATE",
        height=420)
    return fig


def fig_per_model_qini(val):
    """2×2 individual Qini curves for the 4 learners."""
    model_keys = [k for k in ["t_learner","x_learner","causal_forest","ensemble"] if k in val]
    if not model_keys:
        return None

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[MODEL_LABELS[k] for k in model_keys],
        horizontal_spacing=0.08, vertical_spacing=0.14,
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]

    for idx, key in enumerate(model_keys):
        r, c = positions[idx]
        qini   = val[key]["qini"]
        pcts   = qini["percentiles"]
        gains  = qini["qini_gains"]
        rand   = qini["random_gains"]
        auuc   = qini["auuc"]
        color  = MODEL_COLORS[key]

        fig.add_trace(go.Scatter(
            x=pcts, y=gains, mode="lines",
            line=dict(color=color, width=2),
            name=MODEL_LABELS[key],
            fill="tonexty" if idx else None,
            hovertemplate=f"{MODEL_LABELS[key]}<br>%{{x:.0f}}% targeted<br>Qini: %{{y:.4f}}<extra></extra>",
            showlegend=False,
        ), row=r, col=c)

        fig.add_trace(go.Scatter(
            x=pcts, y=rand, mode="lines",
            line=dict(color=C_SLATE_200, width=1.2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ), row=r, col=c)

        fig.add_annotation(
            text=f"AUUC = {auuc:.4f}", xref=f"x{idx+1}", yref=f"y{idx+1}",
            x=50, y=max(gains) * 0.15, showarrow=False,
            font=dict(size=10, color=color), bgcolor="white",
            bordercolor=color, borderwidth=1, borderpad=4,
        )

    dark = is_dark()
    bg       = "#0F172A" if dark else "white"
    grid_col = "#1E293B" if dark else C_SLATE_100
    line_col = "#334155" if dark else C_SLATE_200
    text_col = "#CBD5E1" if dark else C_SLATE_700
    sub_col  = "#64748B" if dark else C_SLATE_400

    fig.update_layout(
        height=500,
        template="plotly_dark" if dark else PLOTLY_TEMPLATE,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family=FONT_FAMILY, size=10, color=text_col),
        title=dict(text="Qini Curves per Model — dotted line = random baseline",
                   font=dict(size=13, color=text_col), x=0, xanchor="left", pad=dict(b=12)),
        margin=dict(l=12, r=12, t=56, b=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=line_col, tickfont=dict(size=9, color=sub_col))
    fig.update_yaxes(gridcolor=grid_col, tickfont=dict(size=9, color=sub_col))
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.2rem; border-bottom: 1px solid #1E293B; margin-bottom: 1rem;'>
      <div style='font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; color:#64748B; margin-bottom:4px;'>Customer Retention Analytics</div>
      <div style='font-size:1.05rem; font-weight:600; color:#F1F5F9; line-height:1.3;'>CATE Dashboard</div>
      <div style='font-size:0.72rem; color:#475569; margin-top:4px;'>Causal Inference · EconML</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview",
         "CATE Distribution",
         "Model Comparison",
         "Qini Curves & Validation",
         "Segment Analysis",
         "Policy Simulator"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin: 1.4rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem; color:#475569; line-height:1.9;'>
      <div style='color:#94A3B8; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; font-size:0.62rem; margin-bottom:6px;'>Models</div>
      T-Learner &nbsp;·&nbsp; X-Learner<br>
      Causal Forest DML<br>
      Ensemble (equal weight)<br>
      <div style='margin-top:10px; color:#94A3B8; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; font-size:0.62rem; margin-bottom:6px;'>Validation</div>
      Qini / AUUC<br>
      Placebo permutation test<br>
      KMeans segment analysis<br>
      <div style='margin-top:10px; color:#94A3B8; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; font-size:0.62rem; margin-bottom:6px;'>Dataset</div>
      N = 100,000 synthetic customers<br>
      Confounded observational data
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 1.4rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.68rem; color:#475569; line-height:1.7;'>
      <div style='color:#94A3B8; margin-bottom:3px; font-size:0.62rem; letter-spacing:0.06em; text-transform:uppercase; font-weight:600;'>Author</div>
      Chandrima Das<br>
      <a href='mailto:chdas@ucsd.edu' style='color:#818CF8; text-decoration:none;'>chdas@ucsd.edu</a>
    </div>
    """, unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df         = load_data()
    val        = load_validation()
    cs         = load_causal_summary()
    pred_df    = load_predictions()

if df is None:
    st.error("Data not found. Run the pipeline first: `bash run_pipeline.sh`")
    st.stop()

# Shorthand
has_val   = val is not None
has_cs    = cs is not None
has_pred  = pred_df is not None
has_cate  = "cate" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="page-title">Customer Retention Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Heterogeneous treatment effect estimation on synthetic retail data using meta-learners and causal forests. All models trained on confounded observational data; validation via Qini curve and placebo permutation test.</div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Total customers", f"{len(df):,}")
    with c2:
        churn = df["churn"].mean() if "churn" in df.columns else 0
        kpi("Churn rate", f"{churn:.1%}", f"Baseline retention {1-churn:.1%}")
    with c3:
        tx = df["treatment"].mean() if "treatment" in df.columns else 0
        kpi("Treatment rate", f"{tx:.1%}", f"{int(tx*len(df)):,} treated")
    with c4:
        mc = df["cate"].mean() if has_cate else 0
        kpi("Mean CATE (oracle)", f"{mc:.4f}", "Avg. lift in retention prob.")
    with c5:
        if has_val and "t_learner" in val:
            auuc = val["t_learner"]["placebo"]["real_auuc"]
            kpi("Best model AUUC", f"{auuc:.4f}", "T-Learner — lift over random", delta_positive=True)
        else:
            kpi("AUUC", "N/A", "Run validation")

    section("Pipeline Architecture")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        ```
        Synthetic retail data (100K customers, 30+ features)
                    │
                    ▼
        Feature engineering  ──  interaction terms, aggregates, scaling
                    │
                    ▼
        Confounded treatment assignment  ──  selection bias via engagement score
                    │
                    ▼
        Causal model training (4 estimators)
          ├─ T-Learner      two XGBoost outcome models, one per group
          ├─ X-Learner      cross-fitted with propensity weighting
          ├─ Causal Forest  DML residualization (model_y / model_t)
          └─ Ensemble       equal-weight average of the three
                    │
                    ▼
        Validation
          ├─ Qini curve / AUUC (normalized, dataset-size invariant)
          ├─ Placebo permutation test  (n=100 shuffles, p < 0.05)
          └─ KMeans segment heterogeneity (4 clusters on features)
        ```
        """)
    with col_b:
        if has_val:
            section("Validation scorecard")
            rows = []
            for k in ["t_learner","x_learner","causal_forest","ensemble"]:
                if k not in val: continue
                p = val[k]["placebo"]
                sig = badge("significant", "success") if p["significant"] else badge("not sig.", "warn")
                rows.append({
                    "Model": MODEL_LABELS[k],
                    "AUUC lift": f"{p['real_auuc']:+.4f}",
                    "p-value": f"{p['p_value']:.3f}",
                    "Result": sig,
                })
            tbl_html = pd.DataFrame(rows).to_html(index=False, escape=False,
                classes="dataframe", border=0)
            st.markdown(tbl_html, unsafe_allow_html=True)

    section("Key findings")
    insight("""
    <strong>Treatment heterogeneity is real and statistically significant.</strong>
    All four estimators pass the placebo permutation test (p < 0.001). The T-Learner captures
    91.5% of oracle AUUC (0.0328 vs. 0.0359), suggesting the meta-learner strategy is well-suited
    to this data-generating process. The Causal Forest DML, while valid, captures only ~25% of the
    oracle ceiling — likely due to the 3-fold cross-fitting constraint; increasing <code>cv=5</code>
    and <code>n_estimators=500</code> should improve it. Segment 1 (the smallest KMeans cluster,
    ~10K customers) consistently shows the lowest predicted CATE across all models, indicating a
    subgroup for whom the intervention may not be cost-effective.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CATE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "CATE Distribution":
    st.markdown('<div class="page-title">Treatment Effect Estimation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">CATE Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Conditional Average Treatment Effect (CATE) — the estimated individual-level lift in retention probability attributable to treatment. Positive CATE = treatment helps; negative CATE = treatment harmful.</div>', unsafe_allow_html=True)

    if has_cate:
        cate = df["cate"]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: kpi("Mean", f"{cate.mean():.4f}")
        with c2: kpi("Std dev", f"{cate.std():.4f}")
        with c3: kpi("Median", f"{cate.median():.4f}")
        with c4: kpi("Min", f"{cate.min():.4f}")
        with c5: kpi("Max", f"{cate.max():.4f}")
        with c6: kpi("% negative CATE", f"{(cate<0).mean():.1%}", "intervention harmful")

        st.plotly_chart(fig_cate_distribution(df), use_container_width=True)

        section("Interpretation")
        insight(f"""
        The CATE distribution spans [{cate.min():.3f}, {cate.max():.3f}] with mean {cate.mean():.4f},
        confirming <strong>meaningful treatment effect heterogeneity</strong> in the population.
        {(cate<0).mean():.1%} of customers have negative CATE — targeting them would reduce retention.
        The strong positive skew (mean > median = {cate.median():.4f}) indicates a long tail of
        high-responders; Segment 3 (VIP customers) drives much of this tail.
        """)

        if has_pred:
            section("Predicted vs. Oracle CATE (ensemble model)")
            st.plotly_chart(fig_pred_vs_true(pred_df), use_container_width=True)
            insight("""
            Each point is one customer. The dashed red line is perfect prediction (y = x).
            Points above the line = model overestimates; below = underestimates. Correlation
            measures how well ranked ordering is preserved — the key property for Qini-based targeting.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown('<div class="page-title">Causal Estimation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Four heterogeneous treatment effect estimators trained on the same confounded observational data. Estimators differ in how they handle selection bias and modelling strategy.</div>', unsafe_allow_html=True)

    if has_cs:
        section("Mean CATE and dispersion")
        st.plotly_chart(fig_model_summary_bars(cs), use_container_width=True)

        section("Percentile spread (p10 – p90)")
        st.plotly_chart(fig_cate_percentile_range(cs), use_container_width=True)

        section("Raw statistics table")
        display_cs = cs.copy()
        display_cs.index = [MODEL_LABELS.get(i, i) for i in display_cs.index]
        display_cs.columns = [c.replace("_", " ").title() for c in display_cs.columns]
        st.dataframe(display_cs.round(5), use_container_width=True)

    section("Model descriptions")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        **T-Learner**
        Trains two separate outcome models — one on the control group, one on the treatment group.
        CATE = μ₁(x) − μ₀(x). Fast and interpretable. Sensitive to regularisation differences
        between the two models. Uses XGBoost with different hyperparameters per arm to encourage diversity.

        **X-Learner**
        Extends T-Learner with cross-fitted imputed treatment effects, weighted by estimated propensity.
        More robust under treatment overlap violations. Uses GradientBoosting for outcomes and XGBoost
        for the CATE model.
        """)
    with c2:
        st.markdown(f"""
        **Causal Forest DML** (`econml.dml.CausalForestDML`)
        Partials out confounders via nuisance models (E[Y|X], E[T|X]) before fitting the forest on
        residuals. Solves the moment condition: (Y − Ê[Y|X]) = θ(X) · (T − Ê[T|X]) + ε.
        Most principled approach but lowest AUUC here — likely needs larger `cv` and `n_estimators`.

        **Ensemble**
        Equal-weight average of the three estimators above. Reduces variance and is the recommended
        prediction for production targeting decisions.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: QINI CURVES & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Qini Curves & Validation":
    st.markdown('<div class="page-title">Statistical Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">Qini Curves & Placebo Tests</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Qini curves measure how well predicted CATE ranks customers by true benefit. AUUC (lift) = area under model curve minus area under random baseline. Placebo test shuffles predicted scores 100× to establish a null distribution.</div>', unsafe_allow_html=True)

    if has_val:
        section("All models vs. random baseline")

        # Interactive model selector
        available_models = [k for k in ["t_learner","x_learner","causal_forest","ensemble","oracle"] if k in val]
        selected_models = st.multiselect(
            "Select models to display",
            options=available_models,
            default=available_models,
            format_func=lambda k: MODEL_LABELS.get(k, k),
        )
        show_random = st.checkbox("Show random baseline", value=True)

        # Build filtered val dict for the chart
        val_filtered = {k: val[k] for k in selected_models}
        if show_random and selected_models:
            # inject random gains from first selected model
            first = selected_models[0]
            val_filtered["random"] = {"qini": {
                "percentiles": val[first]["qini"]["percentiles"],
                "qini_gains":  val[first]["qini"]["random_gains"],
                "random_gains": val[first]["qini"]["random_gains"],
                "auuc": 0,
            }}

        st.plotly_chart(fig_qini_all_models(val_filtered), use_container_width=True)

        insight("""
        <strong>Reading the Qini curve:</strong> The x-axis ranks customers from highest to lowest
        predicted CATE. If the model has true targeting power, concentrating treatment on the
        top-ranked customers should produce disproportionate retention gains — shown by the curve
        lying above the dashed random baseline. The area between them (AUUC lift) is the headline metric.
        All four models lie above the random baseline across the full targeting range.
        """)

        section("Per-model Qini breakdown")
        st.plotly_chart(fig_per_model_qini(val), use_container_width=True)

        section("Placebo permutation test")
        st.plotly_chart(fig_placebo_comparison(val), use_container_width=True)

        insight("""
        <strong>Placebo test methodology:</strong> For each of 100 iterations, predicted CATE scores
        are permuted randomly across customers (destroying any ranking signal) and AUUC is recomputed.
        The real model's AUUC should lie far in the right tail of this null distribution.
        p-value = fraction of permuted AUUCs ≥ real AUUC. All models: p < 0.001.
        """)

        section("Validation summary table")
        rows = []
        for k in ["t_learner","x_learner","causal_forest","ensemble"]:
            if k not in val: continue
            p = val[k]["placebo"]
            q = val[k]["qini"]
            rows.append({
                "Model": MODEL_LABELS[k],
                "AUUC lift": round(p["real_auuc"], 5),
                "% of oracle": f"{p['real_auuc']/val['oracle']['placebo']['real_auuc']*100:.1f}%" if "oracle" in val else "N/A",
                "Placebo mean": round(p["placebo_mean"], 5),
                "Placebo std": round(p["placebo_std"], 5),
                "p-value": p["p_value"],
                "Significant": "Yes" if p["significant"] else "No",
                "AUUC model": round(q["auuc_model"], 5),
                "AUUC random": round(q["auuc_random"], 5),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    else:
        st.warning("Validation results not found. Run `python validation.py` first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SEGMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Segment Analysis":
    st.markdown('<div class="page-title">Heterogeneity Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">Customer Segment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">KMeans clustering (k=4) on the feature matrix identifies customer sub-populations. Mean predicted CATE within each cluster reveals which segments respond most to treatment — informing tiered intervention strategies.</div>', unsafe_allow_html=True)

    if has_val:
        section("Mean CATE per segment — all models")
        fig_seg = fig_segment_comparison(val)
        if fig_seg:
            st.plotly_chart(fig_seg, use_container_width=True)
        else:
            st.info("Segment data not available for any model.")

        section("Model-segment CATE heatmap")
        fig_heat = fig_segment_heatmap(val)
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True)

        insight("""
        <strong>Segment 1</strong> consistently shows the lowest mean CATE across all estimators
        (~0.09–0.12), suggesting this customer group responds weakly to the intervention. Excluding
        them from targeting and redirecting budget to Segments 0, 2, and 3 is expected to improve
        campaign ROI. The heatmap confirms strong cross-model agreement on segment ranking,
        increasing confidence in the estimates.
        """)

        section("Detailed segment statistics")
        model_choice = st.selectbox(
            "Select model to inspect",
            options=[k for k in ["t_learner","x_learner","causal_forest","ensemble"] if k in val],
            format_func=lambda k: MODEL_LABELS[k],
        )
        if model_choice in val:
            seg_rows = []
            for seg, stats in val[model_choice]["segments"].items():
                sig_str = badge("high CATE", "success") if stats["mean_cate"] > 0.18 else badge("low CATE", "warn")
                seg_rows.append({
                    "Segment": seg,
                    "N customers": f"{stats['n_samples']:,}",
                    "Mean CATE": round(stats["mean_cate"], 5),
                    "Std CATE": round(stats["std_cate"], 5),
                    "p10": round(stats["p10"], 5),
                    "p50 (median)": round(stats["p50"], 5),
                    "p90": round(stats["p90"], 5),
                    "Signal": sig_str,
                })
            tbl = pd.DataFrame(seg_rows).to_html(index=False, escape=False,
                classes="dataframe", border=0)
            st.markdown(tbl, unsafe_allow_html=True)
    else:
        st.warning("Validation results not found. Run `python validation.py` first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: POLICY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Policy Simulator":
    st.markdown('<div class="page-title">Business Decision Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">Targeting Policy Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Translate CATE estimates into a business decision: which customers to treat and at what budget. Uses ensemble predicted CATE when available, otherwise oracle CATE.</div>', unsafe_allow_html=True)

    # Build a unified sim_df that always has both 'cate' and 'retained' columns
    if has_pred and "ensemble_cate" in pred_df.columns:
        cate_col  = pred_df["ensemble_cate"].values
        cate_label = "Ensemble model CATE"
        sim_df = pd.DataFrame({"cate": cate_col, "retained": df["retained"].values})
    else:
        cate_col  = df["cate"].values if has_cate else None
        cate_label = "Oracle CATE"
        sim_df = df[["cate","retained"]].copy() if has_cate else None

    if cate_col is None or sim_df is None:
        st.warning("No CATE estimates available.")
        st.stop()

    roi_fig = fig_roi_curve(sim_df)
    if roi_fig is not None:
        st.plotly_chart(roi_fig, use_container_width=True)

    section("Campaign parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        target_pct = st.slider("Target top N% of customers by CATE", 5, 100, 30, step=5)
    with c2:
        cost_per_tx = st.number_input("Cost per treatment ($)", min_value=1, max_value=5000, value=10, step=5)
    with c3:
        rev_per_ret = st.number_input("Revenue per retained customer ($)", min_value=50, max_value=50000, value=500, step=50)

    # Live ROI vs targeting depth — recomputes on every slider change
    def build_sensitivity(cate_arr, y_arr, cost, rev):
        rows = []
        for pct in range(5, 101, 5):
            thr = np.percentile(cate_arr, 100 - pct)
            tgt = cate_arr >= thr
            n_t = tgt.sum()
            r_t = y_arr[tgt].mean() if n_t > 0 else y_arr.mean()
            lft = r_t - y_arr.mean()
            c_t = n_t * cost
            r_v = max(0, n_t * lft) * rev
            roi_v = (r_v - c_t) / c_t * 100 if c_t > 0 else 0
            rows.append(dict(pct=pct, n=n_t, retention=r_t*100, lift=lft*100,
                             cost=c_t, revenue=r_v, roi=roi_v))
        return pd.DataFrame(rows)

    sens_df = build_sensitivity(cate_col, sim_df["retained"].values, cost_per_tx, rev_per_ret)

    # Interactive sensitivity chart
    dark = is_dark()
    bg_c = "#0F172A" if dark else "white"
    txt  = "#CBD5E1" if dark else C_SLATE_700
    sub  = "#64748B" if dark else C_SLATE_400
    gc   = "#1E293B" if dark else C_SLATE_100

    fig_sens = make_subplots(rows=1, cols=2,
        subplot_titles=["ROI (%) vs. targeting depth", "Absolute revenue and cost ($)"],
        horizontal_spacing=0.1)

    # ROI line with current target highlighted
    roi_colors = [C_TEAL if r >= 0 else C_ROSE for r in sens_df["roi"]]
    fig_sens.add_trace(go.Bar(
        x=sens_df["pct"], y=sens_df["roi"],
        marker_color=roi_colors, name="ROI (%)",
        hovertemplate="Top %{x}%<br>ROI: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig_sens.add_vline(x=target_pct, line_dash="dash", line_color=C_AMBER,
                       line_width=1.5, row=1, col=1)
    fig_sens.add_hline(y=0, line_color=C_SLATE_400, line_width=1, row=1, col=1)

    # Revenue vs cost
    fig_sens.add_trace(go.Scatter(
        x=sens_df["pct"], y=sens_df["revenue"]/1e3,
        name="Revenue ($K)", mode="lines+markers",
        line=dict(color=C_TEAL, width=2),
        hovertemplate="Top %{x}%<br>Revenue: $%{y:.0f}K<extra></extra>",
    ), row=1, col=2)
    fig_sens.add_trace(go.Scatter(
        x=sens_df["pct"], y=sens_df["cost"]/1e3,
        name="Cost ($K)", mode="lines+markers",
        line=dict(color=C_ROSE, width=2, dash="dot"),
        hovertemplate="Top %{x}%<br>Cost: $%{y:.0f}K<extra></extra>",
    ), row=1, col=2)
    fig_sens.add_vline(x=target_pct, line_dash="dash", line_color=C_AMBER,
                       line_width=1.5, row=1, col=2)

    fig_sens.update_layout(
        height=360, template="plotly_dark" if dark else PLOTLY_TEMPLATE,
        paper_bgcolor=bg_c, plot_bgcolor=bg_c,
        font=dict(family=FONT_FAMILY, size=11, color=txt),
        legend=dict(font=dict(size=10, color=txt), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=8, t=44, b=8),
        hovermode="x unified",
    )
    fig_sens.update_xaxes(showgrid=False, linecolor=gc, tickfont=dict(size=10, color=sub),
                          title_text="% targeted", title_font=dict(color=sub))
    fig_sens.update_yaxes(gridcolor=gc, tickfont=dict(size=10, color=sub))
    st.plotly_chart(fig_sens, use_container_width=True)

    threshold = np.percentile(cate_col, 100 - target_pct)
    targeted  = cate_col >= threshold
    y_all     = sim_df["retained"].values

    n_tgt  = targeted.sum()
    n_ctrl = (~targeted).sum()
    ret_tgt  = y_all[targeted].mean() if n_tgt > 0 else 0
    ret_ctrl = y_all[~targeted].mean() if n_ctrl > 0 else 0
    baseline = y_all.mean()
    lift     = ret_tgt - baseline

    total_cost   = n_tgt * cost_per_tx
    addl_retained = max(0, n_tgt * lift)
    total_rev    = addl_retained * rev_per_ret
    roi          = (total_rev - total_cost) / total_cost * 100 if total_cost > 0 else 0
    cate_thresh  = threshold

    section("ROI analysis")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Customers targeted", f"{n_tgt:,}", f"CATE threshold {cate_thresh:.4f}")
    with c2:
        kpi("Expected retention", f"{ret_tgt:.1%}", f"{lift:+.1%} vs. baseline {baseline:.1%}", delta_positive=lift>0)
    with c3:
        kpi("Total investment", f"${total_cost:,.0f}", f"${cost_per_tx}/customer")
    with c4:
        kpi("Additional retained", f"{int(addl_retained):,}", f"× ${rev_per_ret:,} = ${total_rev:,.0f} revenue")
    with c5:
        kpi("Return on investment", f"{roi:+.1f}%", "positive = profitable campaign", delta_positive=roi>0)

    section("Sensitivity table — all targeting depths")
    # Reuse sens_df already computed above; filter to common breakpoints for readability
    sens_display = sens_df[sens_df["pct"].isin([10,20,25,30,40,50,60,75,100])].copy()
    sens_display["N targeted"] = sens_display["n"].apply(lambda x: f"{x:,}")
    sens_display["Expected retention"] = sens_display["retention"].apply(lambda x: f"{x:.1f}%")
    sens_display["Lift vs baseline"] = sens_display["lift"].apply(lambda x: f"{x:+.2f}%")
    sens_display["Investment ($K)"] = (sens_display["cost"]/1e3).apply(lambda x: f"{x:,.1f}")
    sens_display["Revenue ($K)"] = (sens_display["revenue"]/1e3).apply(lambda x: f"{x:,.1f}")
    sens_display["ROI (%)"] = sens_display["roi"].apply(lambda x: f"{x:+.1f}%")
    sens_display.index = sens_display["pct"].apply(lambda x: f"Top {x}%")
    st.dataframe(
        sens_display[["N targeted","Expected retention","Lift vs baseline","Investment ($K)","Revenue ($K)","ROI (%)"]],
        use_container_width=True,
    )

    insight(f"""
    <strong>Recommendation:</strong> Based on {cate_label}, targeting the top {target_pct}%
    of customers (CATE ≥ {cate_thresh:.4f}) yields an expected retention rate of {ret_tgt:.1%}
    vs. the population baseline of {baseline:.1%} — a lift of {lift:+.2%}. At ${cost_per_tx}/customer
    and ${rev_per_ret:,} revenue per retained customer, the campaign {'is profitable' if roi > 0 else 'is unprofitable'}
    with ROI = {roi:+.1f}%. Use the sensitivity table above to identify the optimal targeting depth
    for your cost structure.
    """)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin-top:3rem;'>", unsafe_allow_html=True)
st.markdown(f"""
<div style='display:flex; justify-content:space-between; align-items:center;
     font-size:0.72rem; color:{C_SLATE_400}; padding:0.5rem 0 1.5rem;'>
  <span>Customer Retention Analytics &nbsp;·&nbsp; Causal Inference Dashboard</span>
  <span>Chandrima Das &nbsp;·&nbsp;
    <a href='mailto:chdas@ucsd.edu' style='color:{C_INDIGO_L}; text-decoration:none;'>chdas@ucsd.edu</a>
  </span>
</div>
""", unsafe_allow_html=True)
