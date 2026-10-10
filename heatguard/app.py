"""
app.py
======
HEATGUARD AI  -  ANN-Based Heat Wave Prediction for May (Mumbai)
Soft Computing academic project.

Single-file Streamlit application with four pages:
    Heat Wave  |  ANN Simulation  |  Model  |  Methodology

Every model number shown in this UI is produced by ann.py at run time.
Nothing is hard-coded.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ann as A
import config as cfg
import data_utils as du
import heatwave as hw

# ==========================================================================
# 0. PAGE CONFIG + THEME
# ==========================================================================
st.set_page_config(page_title="HEATGUARD AI", page_icon="🔥",
                   layout="wide", initial_sidebar_state="collapsed")

BG, PANEL, BORDER = cfg.BG, cfg.PANEL, cfg.BORDER
TEXT, MUTED, ACCENT = cfg.TEXT, cfg.MUTED, cfg.ACCENT

CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp { background: #0a0e14; color: #e6edf3; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3, h4 { color: #e6edf3; letter-spacing: -0.01em; }

.hg-brand { font-size: 2.0rem; font-weight: 700; letter-spacing: 0.16em;
            color: #e6edf3; margin: 0; line-height: 1.1; }
.hg-sub { color: #8b98a9; font-size: 0.92rem; letter-spacing: 0.04em; margin-top: .25rem; }
.hg-place { color: #38bdf8; font-size: .82rem; letter-spacing: .18em; font-weight: 600; }

.hg-sec { font-size: .76rem; letter-spacing: .22em; color: #8b98a9; font-weight: 600;
          text-transform: uppercase; border-bottom: 1px solid #1e2a3a;
          padding-bottom: .5rem; margin: 1.8rem 0 1rem 0; }

.hg-card { background: #111823; border: 1px solid #1e2a3a; border-radius: 10px;
           padding: 1rem 1.1rem; height: 100%; }
.hg-card .k { color: #8b98a9; font-size: .68rem; letter-spacing: .16em;
              text-transform: uppercase; font-weight: 600; }
.hg-card .v { color: #e6edf3; font-size: 1.85rem; font-weight: 650; line-height: 1.25;
              margin-top: .35rem; font-variant-numeric: tabular-nums; }
.hg-card .u { color: #8b98a9; font-size: .95rem; font-weight: 500; margin-left: .18rem; }
.hg-card .n { color: #8b98a9; font-size: .74rem; margin-top: .3rem; }

.hg-verdict { border-radius: 12px; padding: 1.1rem 1.3rem; border: 1px solid;
              display: flex; align-items: baseline; gap: 1.4rem; flex-wrap: wrap; }
.hg-verdict .big { font-size: 1.5rem; font-weight: 700; letter-spacing: .06em; }
.hg-verdict .small { color: #8b98a9; font-size: .85rem; }

.hg-pill { display: inline-block; padding: .22rem .6rem; border-radius: 999px;
           font-size: .7rem; letter-spacing: .12em; font-weight: 600;
           text-transform: uppercase; border: 1px solid #1e2a3a;
           color: #8b98a9; background: #111823; }
.hg-pill.ok   { color: #34d399; border-color: #1c5142; }
.hg-pill.warn { color: #f59e0b; border-color: #5a4212; }

.hg-mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .84rem;
           color: #cbd5e1; background: #0d141d; border: 1px solid #1e2a3a;
           border-radius: 8px; padding: .75rem .9rem; white-space: pre-wrap; }
.hg-note { color: #8b98a9; font-size: .8rem; line-height: 1.55; }

div[data-testid="stHorizontalBlock"] .stButton > button {
  width: 100%; border-radius: 8px; font-weight: 600; letter-spacing: .04em;
  border: 1px solid #1e2a3a; background: #111823; color: #8b98a9;
  transition: all .15s ease; padding: .5rem .2rem; }
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
  border-color: #38bdf8; color: #e6edf3; }
div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
  background: #16212f; color: #e6edf3; border-color: #38bdf8;
  box-shadow: inset 0 -2px 0 0 #38bdf8; }
.stButton > button { border-radius: 8px; }

@keyframes hgFade { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.hg-fade { animation: hgFade .35s ease-out both; }

div[data-testid="stMetricValue"] { font-size: 1.4rem; }
hr { border-color: #1e2a3a; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=45, r=20, t=45, b=40),
)


# ==========================================================================
# 1. SMALL UI HELPERS
# ==========================================================================
def section(title: str) -> None:
    st.markdown(f'<div class="hg-sec">{title}</div>', unsafe_allow_html=True)


def card(key: str, value, unit: str = "", note: str = "", color: str | None = None) -> str:
    col = f"color:{color};" if color else ""
    return (f'<div class="hg-card"><div class="k">{key}</div>'
            f'<div class="v" style="{col}">{value}<span class="u">{unit}</span></div>'
            f'<div class="n">{note}</div></div>')


def mono(text: str) -> None:
    st.markdown(f'<div class="hg-mono">{text}</div>', unsafe_allow_html=True)


# ==========================================================================
# 2. CACHED DATA + MODEL  (trained once, reused on every rerun)
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_data():
    return du.load_dataset()


@st.cache_resource(show_spinner="Training the ANN (runs once per session)...")
def build_model(_df: pd.DataFrame):
    """
    Preprocess -> train the ANN -> evaluate.

    @st.cache_resource means the network is trained ONCE. Moving a slider or
    changing page never retrains it.
    """
    prep = du.prepare_training_data(_df, seed=cfg.SEED)
    net = A.SimpleANN(n_input=len(du.FEATURES), n_hidden=cfg.N_HIDDEN,
                      n_output=1, seed=cfg.SEED)
    net.fit(prep["X_train"], prep["y_train"],
            epochs=cfg.TRAIN_EPOCHS, learning_rate=cfg.TRAIN_LR,
            X_val=prep["X_val"], y_val=prep["y_val"], record_every=10)
    metrics = {s: A.evaluate(prep[f"y_{s}"], net.predict_proba(prep[f"X_{s}"]))
               for s in ("train", "val", "test")}
    return net, prep, metrics


# ==========================================================================
# 3. PLOTLY FIGURES
# ==========================================================================
def gauge_figure(score: float, height: int = 320) -> go.Figure:
    """Large risk gauge. The needle position IS the ANN output."""
    level = hw.risk_level(score)
    colour = hw.risk_color(level)
    lo, mid = cfg.RISK_LOW_MAX * 100, cfg.RISK_MODERATE_MAX * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score * 100.0,
        number={"suffix": "%", "font": {"size": 62, "color": colour}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": MUTED,
                     "tickfont": {"size": 10, "color": MUTED},
                     "tickvals": [0, lo, mid, 100]},
            "bar": {"color": colour, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [{"range": [0, lo], "color": "rgba(52,211,153,0.10)"},
                      {"range": [lo, mid], "color": "rgba(245,158,11,0.10)"},
                      {"range": [mid, 100], "color": "rgba(239,68,68,0.12)"}],
            "threshold": {"line": {"color": MUTED, "width": 2}, "thickness": 0.82,
                          "value": cfg.DECISION_THRESHOLD * 100},
        }))
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_layout(margin=dict(l=25, r=25, t=15, b=5))
    return fig


# ---- map: Plotly renamed Scattermapbox -> Scattermap, support both ---------
_HAS_SCATTERMAP = hasattr(go, "Scattermap")
_MAP_KEY = "map" if _HAS_SCATTERMAP else "mapbox"


def _map_trace(**kwargs):
    return go.Scattermap(**kwargs) if _HAS_SCATTERMAP else go.Scattermapbox(**kwargs)


def map_figure(points: list[dict], style: str | None = None) -> go.Figure:
    """
    Geographic map of the study area, drawn on a token-free OpenStreetMap /
    Carto basemap (no API key, no account).

    `points` is a list of dicts with keys: name, lat, lon, score, level, popup.
    Marker COLOUR comes from the risk band and marker SIZE is proportional to
    the ANN score - both are read from the live model output, never hard-coded.

    Only locations that genuinely exist in the data are drawn. With a
    single-location dataset that is one marker labelled "Model Study Area";
    no neighbourhood boundaries or per-suburb percentages are invented.
    """
    fig = go.Figure()

    # Soft halo behind each marker so the point reads at a glance.
    fig.add_trace(_map_trace(
        lat=[p["lat"] for p in points], lon=[p["lon"] for p in points],
        mode="markers",
        marker=dict(size=[26 + 46 * p["score"] for p in points],
                    color=[hw.risk_color(p["level"]) for p in points], opacity=0.22),
        hoverinfo="skip", showlegend=False))

    fig.add_trace(_map_trace(
        lat=[p["lat"] for p in points], lon=[p["lon"] for p in points],
        mode="markers+text",
        marker=dict(size=[13 + 20 * p["score"] for p in points],
                    color=[hw.risk_color(p["level"]) for p in points], opacity=0.95),
        text=[p["name"] for p in points], textposition="top right",
        textfont=dict(size=11, color=hw.risk_color(points[0]["level"])),
        hoverinfo="text", hovertext=[p["popup"] for p in points], showlegend=False))

    centre = dict(lat=float(np.mean([p["lat"] for p in points])),
                  lon=float(np.mean([p["lon"] for p in points])))
    fig.update_layout(**{_MAP_KEY: dict(style=style or cfg.MAP_STYLE,
                                        center=centre, zoom=cfg.STUDY_AREA["zoom"])})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=cfg.MAP_HEIGHT,
                      margin=dict(l=0, r=0, t=0, b=0),
                      font=dict(color=TEXT, family="Inter, system-ui, sans-serif"))
    return fig


def network_figure(net: A.SimpleANN, activations: dict | None = None,
                   stage: int = 4, selected: int | None = None,
                   height: int = 470) -> go.Figure:
    """
    Draw the 6 -> 8 -> 1 architecture.

    Edge colour = sign of the weight (orange = positive / excitatory,
    blue = negative / inhibitory). Edge width = |weight|.
    `stage` (0..4) drives the restrained step-by-step animation.
    `activations` lights the neurons with their real activation values.
    """
    in_names = [du.FEATURE_META[f]["short"] for f in du.FEATURES]
    n_in, n_hid = net.n_input, net.n_hidden

    def ys(n):
        return list(np.linspace(0.92, 0.08, n)) if n > 1 else [0.5]

    y_in, y_hid, y_out = ys(n_in), ys(n_hid), [0.5]
    x_in, x_hid, x_out = 0.0, 1.0, 2.0

    fig = go.Figure()
    max_w1 = max(np.abs(net.W1).max(), 1e-9)
    max_w2 = max(np.abs(net.W2).max(), 1e-9)

    def edge_style(w, wmax, lit):
        base = (249, 146, 54) if w >= 0 else (56, 189, 248)
        alpha = (0.12 + 0.55 * abs(w) / wmax) if lit else 0.05
        return f"rgba({base[0]},{base[1]},{base[2]},{alpha:.3f})", 0.6 + 3.0 * abs(w) / wmax

    for i in range(n_in):
        for j in range(n_hid):
            w = net.W1[i, j]
            lit = stage >= 1 and (selected is None or selected == j)
            colour, width = edge_style(w, max_w1, lit)
            fig.add_trace(go.Scatter(x=[x_in, x_hid], y=[y_in[i], y_hid[j]], mode="lines",
                                     line=dict(color=colour, width=width), hoverinfo="text",
                                     text=f"{in_names[i]} to H{j+1}: w = {w:+.4f}",
                                     showlegend=False))
    for j in range(n_hid):
        w = net.W2[j, 0]
        lit = stage >= 3 and (selected is None or selected == j)
        colour, width = edge_style(w, max_w2, lit)
        fig.add_trace(go.Scatter(x=[x_hid, x_out], y=[y_hid[j], y_out[0]], mode="lines",
                                 line=dict(color=colour, width=width), hoverinfo="text",
                                 text=f"H{j+1} to Output: w = {w:+.4f}", showlegend=False))

    def node_trace(xs, ys_, labels, acts, lit, size, textpos):
        if acts is not None and lit:
            marker = dict(size=size, color=acts,
                          colorscale=[[0, "#16212f"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                          cmin=0, cmax=1, line=dict(color=ACCENT, width=1.4))
            hover = [f"{l}<br>activation = {a:.4f}" for l, a in zip(labels, acts)]
        else:
            marker = dict(size=size, color="#1d2b3d" if lit else "#16212f",
                          line=dict(color=ACCENT if lit else BORDER, width=1.4))
            hover = labels
        return go.Scatter(x=xs, y=ys_, mode="markers+text", marker=marker,
                          text=labels, textposition=textpos,
                          textfont=dict(size=10, color=TEXT if lit else MUTED),
                          hoverinfo="text", hovertext=hover, showlegend=False)

    a1 = activations.get("a1") if activations else None
    a2 = activations.get("a2") if activations else None
    x_act = activations.get("x") if activations else None

    fig.add_trace(node_trace([x_in] * n_in, y_in, in_names,
                             list(np.clip(x_act, 0, 1)) if x_act is not None else None,
                             stage >= 0, 26, "middle left"))
    fig.add_trace(node_trace([x_hid] * n_hid, y_hid, [f"H{j+1}" for j in range(n_hid)],
                             list(a1) if (a1 is not None and stage >= 2) else None,
                             stage >= 2, 28, "middle center"))
    fig.add_trace(node_trace([x_out], y_out, ["OUT"],
                             [float(a2)] if (a2 is not None and stage >= 4) else None,
                             stage >= 4, 38, "middle right"))

    for x, label in [(x_in, "INPUT LAYER (6)"), (x_hid, f"HIDDEN LAYER ({n_hid})"),
                     (x_out, "OUTPUT (1)")]:
        fig.add_annotation(x=x, y=1.05, text=label, showarrow=False,
                           font=dict(size=10, color=MUTED), xanchor="center")

    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_xaxes(visible=False, range=[-0.55, 2.45])
    fig.update_yaxes(visible=False, range=[-0.05, 1.15])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), hovermode="closest")
    return fig


def flow_figure() -> go.Figure:
    """The end-to-end project flow, drawn as a clean vertical chain."""
    steps = [("CLIMATE DATA", "May observations: 6 variables + label", ACCENT),
             ("PREPROCESSING", "validate · min–max scale (train stats only)", ACCENT),
             ("TRAIN / VALIDATE / TEST", "stratified split, no leakage", ACCENT),
             ("ANN   6 → 8 → 1", "sigmoid hidden + sigmoid output", ACCENT),
             ("WEIGHT LEARNING", "forward → error → backprop → w − η·∂E/∂w", ACCENT),
             ("NEW CLIMATE INPUT", "the six values a user enters", "#f59e0b"),
             ("ANN PREDICTION", "one forward pass with the learned weights", "#f59e0b"),
             ("HEAT-WAVE SCORE", "sigmoid output in [0, 1]", "#f59e0b"),
             ("MAP + ALERT", "gauge · risk band · study-area marker", "#ef4444")]

    n = len(steps)
    fig = go.Figure()
    box_h, gap = 0.72, 0.28
    for k, (title, sub, colour) in enumerate(steps):
        y = n - k
        fig.add_shape(type="rect", x0=0.06, x1=0.94, y0=y - box_h / 2, y1=y + box_h / 2,
                      line=dict(color=colour, width=1.2), fillcolor="rgba(17,24,35,0.9)",
                      layer="below")
        fig.add_annotation(x=0.10, y=y + 0.13, text=f"<b>{title}</b>", showarrow=False,
                           xanchor="left", font=dict(size=12, color=colour))
        fig.add_annotation(x=0.10, y=y - 0.16, text=sub, showarrow=False,
                           xanchor="left", font=dict(size=10, color=MUTED))
        if k < n - 1:
            fig.add_annotation(x=0.5, y=y - box_h / 2, ax=0.5, ay=y - box_h / 2 - gap,
                               xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1,
                               arrowwidth=1.2, arrowcolor=BORDER, text="")
    fig.update_layout(**PLOT_LAYOUT, height=150 + 58 * n)
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0.2, n + 0.8])
    fig.update_layout(margin=dict(l=5, r=5, t=10, b=10))
    return fig


# ==========================================================================
# 4. BOOTSTRAP: load data, train model, guard against errors
# ==========================================================================
try:
    DF, META = load_data()
except du.DatasetError as exc:
    st.markdown('<div class="hg-brand">HEATGUARD AI</div>', unsafe_allow_html=True)
    st.error(f"**Dataset problem.**\n\n{exc}")
    st.info("Expected file: `data/mumbai_may_weather.csv` with columns:\n\n"
            "`" + "`, `".join(du.REQUIRED_COLUMNS) + "`\n\n"
            "Delete the file to let the app regenerate a clearly-marked demo dataset.")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected problem while loading the dataset: {exc}")
    st.stop()

try:
    NET, PREP, METRICS = build_model(DF)
except Exception as exc:
    st.error(f"The ANN could not be trained on this dataset: {exc}")
    st.stop()

SCALER = PREP["scaler"]
MODEL_READY = NET.is_trained


# ==========================================================================
# 5. PREDICTION HELPER
# ==========================================================================
def predict_from_raw(values: dict) -> dict:
    """Raw climate values -> training scaler -> real forward pass -> everything."""
    raw = np.array([[float(values[f]) for f in du.FEATURES]])
    x = SCALER.transform(raw)
    cache = NET.forward(x)
    score = float(cache["a2"][0, 0])
    return {"raw": raw[0], "x": x[0], "cache": cache, "score": score,
            "values": dict(values), "level": hw.risk_level(score),
            "verdict": "HEAT WAVE" if score >= cfg.DECISION_THRESHOLD else "NO HEAT WAVE"}


def study_area_points(result: dict) -> list[dict]:
    """
    Build the marker list for the map from the CURRENT model state.

    Single-location dataset  -> one "Model Study Area" marker.
    Dataset with real station coordinates -> one marker per station, each with
    that station's own ANN prediction for its most recent record.
    """
    v = result["values"]
    unit = {f: du.FEATURE_META[f]["unit"] for f in du.FEATURES}

    def popup(name, score, level, vals, extra=""):
        return (f"<b>{name}</b><br>"
                f"Heat-wave score: <b>{score*100:.1f}%</b><br>"
                f"Status: <b>{level} RISK</b> — "
                f"{'HEAT WAVE' if score >= cfg.DECISION_THRESHOLD else 'NO HEAT WAVE'}<br>"
                f"Temperature: {vals['max_temp']:.1f} {unit['max_temp']}<br>"
                f"Humidity: {vals['humidity']:.0f} {unit['humidity']}<br>"
                f"Wind: {vals['wind_speed']:.1f} {unit['wind_speed']}<br>"
                f"Rainfall: {vals['rainfall']:.1f} {unit['rainfall']}" + extra)

    if META.get("has_station_geography") and META.get("stations") is not None:
        pts = []
        for _, row in META["stations"].iterrows():
            sub = DF[DF["station"] == row["station"]]
            if len(sub) == 0:
                continue
            last = sub.iloc[-1]
            vals = {f: float(last[f]) for f in du.FEATURES}
            s = float(NET.predict_proba(SCALER.transform(
                np.array([[vals[f] for f in du.FEATURES]])))[0, 0])
            lvl = hw.risk_level(s)
            pts.append({"name": str(row["station"]), "lat": float(row["latitude"]),
                        "lon": float(row["longitude"]), "score": s, "level": lvl,
                        "popup": popup(str(row["station"]), s, lvl, vals,
                                       f"<br>Observed label: {int(last[du.TARGET])}"
                                       f"<br>Record date: {last['date']}")})
        if pts:
            return pts

    return [{"name": "MODEL STUDY AREA", "lat": cfg.STUDY_AREA["lat"],
             "lon": cfg.STUDY_AREA["lon"], "score": result["score"],
             "level": result["level"],
             "popup": popup(cfg.STUDY_AREA["label"], result["score"],
                            result["level"], v)}]


# ==========================================================================
# 6. HEADER + NAVBAR
# ==========================================================================
PAGES = ["🔥 Heat Wave", "🧠 ANN Simulation", "📊 Model", "ℹ️ Methodology"]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

head_l, head_r = st.columns([3, 2], vertical_alignment="center")
with head_l:
    st.markdown('<div class="hg-brand">HEATGUARD AI</div>'
                '<div class="hg-sub">ANN-Based Heat-Wave Prediction System</div>',
                unsafe_allow_html=True)
with head_r:
    demo_pill = ('<span class="hg-pill warn">demo data</span>' if META["is_demo"]
                 else '<span class="hg-pill ok">real dataset</span>')
    st.markdown(
        f'<div style="text-align:right">'
        f'<span class="hg-place">{cfg.STUDY_AREA["name"].upper()} &nbsp;•&nbsp; '
        f'{cfg.TARGET_MONTH.upper()}</span><br>'
        f'<span class="hg-pill ok">{"ANN trained" if MODEL_READY else "model not trained"}'
        f'</span> {demo_pill} <span class="hg-pill">{META["rows"]} records</span>'
        f'</div>', unsafe_allow_html=True)

st.write("")
nav_cols = st.columns(len(PAGES) + 1)
for col, name in zip(nav_cols, PAGES):
    with col:
        if st.button(name, key=f"nav_{name}", width="stretch",
                     type="primary" if st.session_state.page == name else "secondary"):
            # st.rerun() matters here. Each button's "primary" highlight is decided
            # while the navbar is being drawn, which happens BEFORE the click of a
            # later button is processed. Without this rerun the highlighted tab would
            # stay one click behind the page actually shown.
            st.session_state.page = name
            st.rerun()
with nav_cols[-1]:
    PRESENTING = st.toggle("Presentation", key="presentation_mode",
                           help="Hide the working controls and show only the "
                                "demonstration essentials: map, score, climate "
                                "conditions, architecture and key metrics.")
st.write("")

if META["is_demo"] and not PRESENTING:
    st.warning(
        "**DEMO DATA.** `data/mumbai_may_weather.csv` is SYNTHETIC data generated by "
        "`data_utils.make_demo_dataframe()` so the application can be run and tested. "
        "These are **not real observations**. Replace the CSV with genuine May records "
        "for Mumbai and delete `data/DEMO_DATA.marker` to remove this banner — nothing "
        "else needs to change.", icon="⚠️")
if META["dropped_rows"] and not PRESENTING:
    st.info(f"{META['dropped_rows']} row(s) with missing or non-numeric values were skipped.")


# ==========================================================================
# 7. PAGE 1 — HEAT WAVE COMMAND CENTRE
# ==========================================================================
def page_heatwave():
    defaults = du.default_inputs(DF)

    # ---------------------------------------------- climate input panel ---
    if not PRESENTING:
        section("Climate Input Panel  —  conditions to evaluate")
        st.markdown('<div class="hg-note">Set the six climate variables, then press '
                    '<b>Predict Heat Wave</b>. The score comes from the trained ANN.</div>',
                    unsafe_allow_html=True)
        st.write("")
        cols = st.columns(3)
        values = {}
        for i, f in enumerate(du.FEATURES):
            m = du.FEATURE_META[f]
            with cols[i % 3]:
                values[f] = st.slider(f"{m['label']}  ({m['unit']})",
                                      float(m["min"]), float(m["max"]),
                                      float(st.session_state.get(f"in_{f}", defaults[f])),
                                      float(m["step"]), key=f"in_{f}")
        b1, b2, _ = st.columns([1, 1, 3])
        with b1:
            go_pred = st.button("⚡  Predict Heat Wave", type="primary",
                                width="stretch", key="btn_predict")
        with b2:
            if st.button("↺  Reset Inputs", width="stretch", key="btn_reset_inputs"):
                for f in du.FEATURES:
                    st.session_state.pop(f"in_{f}", None)
                st.session_state.pop("prediction", None)
                st.rerun()
    else:
        values = {f: float(st.session_state.get(f"in_{f}", defaults[f])) for f in du.FEATURES}
        go_pred = False

    if go_pred or "prediction" not in st.session_state:
        st.session_state.prediction = predict_from_raw(values)
    result = st.session_state.prediction
    score, level = result["score"], result["level"]
    colour = hw.risk_color(level)

    # ========================= SECTION 1 — HERO: MAP  +  RISK =============
    section(f"{cfg.STUDY_AREA['name']} study area  •  {cfg.TARGET_MONTH} heat-wave risk")
    hero_l, hero_r = st.columns([1.15, 1])

    with hero_l:
        points = study_area_points(result)
        try:
            st.plotly_chart(map_figure(points), width="stretch",
                            config={"displayModeBar": False, "scrollZoom": False})
        except Exception:
            st.plotly_chart(map_figure(points, style=cfg.MAP_STYLE_FALLBACK),
                            width="stretch", config={"displayModeBar": False})
        if META.get("has_station_geography"):
            st.caption(f"{len(points)} station(s) with real coordinates found in the "
                       "dataset. Each marker shows that station's own ANN prediction "
                       "for its most recent record.")
        else:
            st.caption("**Model Study Area.** One location is modelled, so the map marks "
                       "the study area — it is **not** a spatially resolved city-wide "
                       "forecast. No neighbourhood boundaries or per-suburb percentages "
                       "are shown, because the dataset contains no station-level "
                       "coordinates. Marker colour = risk band, marker size ∝ ANN score. "
                       "Basemap tiles need an internet connection.")

    with hero_r:
        st.plotly_chart(gauge_figure(score), width="stretch",
                        config={"displayModeBar": False})
        st.markdown(
            f'<div class="hg-fade hg-verdict" style="border-color:{colour}55; '
            f'background:{colour}0f;">'
            f'<div><div class="small">ANN VERDICT</div>'
            f'<div class="big" style="color:{colour}">{result["verdict"]}</div></div>'
            f'<div><div class="small">RISK LEVEL</div>'
            f'<div class="big" style="color:{colour}">{level}</div></div></div>',
            unsafe_allow_html=True)
        st.write("")
        st.markdown(
            f'<div class="hg-card"><div class="k">ANN estimated probability</div>'
            f'<div class="v" style="color:{colour}">{score*100:.1f}<span class="u">%</span></div>'
            f'<div class="n">Sigmoid output of the trained network. This is the '
            f'network\'s score, <b>not</b> a calibrated real-world probability — no '
            f'calibration has been performed. Decision threshold = '
            f'{cfg.DECISION_THRESHOLD:.2f}. Risk bands (LOW &lt; {cfg.RISK_LOW_MAX:.2f} ≤ '
            f'MODERATE &lt; {cfg.RISK_MODERATE_MAX:.2f} ≤ HIGH) are '
            f'<b>project display thresholds</b> set in <code>config.py</code>, not '
            f'official meteorological thresholds.</div></div>',
            unsafe_allow_html=True)

    # ==================== SECTION 2 — CURRENT CLIMATE CONDITIONS =========
    section("Current climate conditions")
    c = st.columns(6)
    icons = {"max_temp": "🌡", "min_temp": "🌙", "humidity": "💧",
             "wind_speed": "💨", "rainfall": "🌧", "cloud_cover": "☁"}
    for col, f in zip(c, du.FEATURES):
        m = du.FEATURE_META[f]
        with col:
            st.markdown(card(f"{icons[f]} {m['short']}", f"{values[f]:.1f}", f" {m['unit']}",
                             f"scaled → {result['x'][du.FEATURES.index(f)]:.3f}"),
                        unsafe_allow_html=True)

    # -------------------------------- presentation mode: short circuit ---
    if PRESENTING:
        section("ANN architecture")
        st.plotly_chart(network_figure(NET, height=330), width="stretch",
                        config={"displayModeBar": False})
        section("Key performance metrics  —  held-out test set")
        m = METRICS["test"]
        e = st.columns(4)
        for col, (name, val) in zip(e, [("Accuracy", m["accuracy"]),
                                        ("Precision", m["precision"]),
                                        ("Recall", m["recall"]), ("F1 score", m["f1"])]):
            with col:
                st.markdown(card(name, f"{val*100:.1f}", " %",
                                 f"{m['n']} test rows, {m['n_positive']} heat-wave"),
                            unsafe_allow_html=True)
        model_status_block()
        return

    # ------------------------------------------- model interpretation ----
    section("Model interpretation")
    a1 = result["cache"]["a1"][0]
    contrib = a1 * NET.W2[:, 0]
    order = np.argsort(-np.abs(contrib))
    top = ", ".join(f"H{j+1} ({contrib[j]:+.3f})" for j in order[:3])
    st.markdown(
        f'<div class="hg-note">The network produced a '
        f'<b style="color:{colour}">{level.lower()}</b> heat-wave score of '
        f'<b>{score*100:.1f}%</b> for this combination of six climate inputs. '
        f'The output neuron adds up eight terms <code>h_j · w_j</code> plus a bias; '
        f'the three largest terms right now come from hidden neurons {top}. '
        f'This is an exact decomposition of the output\'s weighted sum into '
        f'<b>hidden-neuron</b> contributions — it is <b>not</b> a feature-attribution '
        f'claim about the six inputs, and no such method (SHAP, permutation '
        f'importance, …) has been run in this project.</div>', unsafe_allow_html=True)

    with st.expander("Why this prediction?  —  inspect the actual computation"):
        st.markdown('<div class="hg-note">An <b>educational walk-through of the neural '
                    'computation</b>, not a formal explainability method.</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.markdown("**Step 1 — inputs, raw and normalised**")
        st.dataframe(pd.DataFrame({
            "feature": [du.FEATURE_META[f]["label"] for f in du.FEATURES],
            "raw value": [f"{result['raw'][i]:.2f} {du.FEATURE_META[f]['unit']}"
                          for i, f in enumerate(du.FEATURES)],
            "min-max scaled x": np.round(result["x"], 4)}),
            width="stretch", hide_index=True)
        st.markdown("**Step 2 — hidden layer**")
        st.latex(r"net_j = \sum_{i=1}^{6} x_i w_{ij} + b_j \qquad h_j = \sigma(net_j)")
        st.dataframe(pd.DataFrame({
            "hidden neuron": [f"H{j+1}" for j in range(NET.n_hidden)],
            "net_j": np.round(result["cache"]["z1"][0], 4),
            "h_j = sigmoid(net_j)": np.round(a1, 4),
            "w_j (hidden→output)": np.round(NET.W2[:, 0], 4),
            "term h_j · w_j": np.round(contrib, 4)}), width="stretch", hide_index=True)
        st.markdown("**Step 3 — output neuron**")
        z2 = float(result["cache"]["z2"][0, 0])
        mono(f"z  = Σ (h_j · w_j) + b   = {contrib.sum():+.5f} + "
             f"({float(NET.b2[0,0]):+.5f}) = {z2:+.5f}\n"
             f"y  = sigmoid(z)          = 1 / (1 + e^(-({z2:.5f}))) = {score:.5f}\n"
             f"score                    = {score*100:.2f} %   →   {level}")

    # ==================== SECTION 3 — WHAT-IF SIMULATION =================
    section("What-if simulation  —  baseline vs scenario")
    st.markdown('<div class="hg-note">The same trained ANN is run a second time with '
                'modified inputs. No retraining, no second model — just another forward '
                'pass.</div>', unsafe_allow_html=True)
    st.write("")

    wf_cols = st.columns(4)
    what_if = dict(values)
    for col, f in zip(wf_cols, ["max_temp", "humidity", "wind_speed", "rainfall"]):
        m = du.FEATURE_META[f]
        with col:
            what_if[f] = st.slider(f"{m['short']} ({m['unit']})", float(m["min"]),
                                   float(m["max"]), float(values[f]), float(m["step"]),
                                   key=f"wf_{f}")

    wf = predict_from_raw(what_if)
    delta = wf["score"] - score
    d_col = hw.risk_color(wf["level"])

    r1, r2, r3 = st.columns([1, 1, 1.4])
    with r1:
        st.markdown(card("Baseline", f"{score*100:.1f}", " %", f"{level} risk", colour),
                    unsafe_allow_html=True)
    with r2:
        st.markdown(card("Scenario", f"{wf['score']*100:.1f}", " %",
                         f"{wf['level']} risk", d_col), unsafe_allow_html=True)
    with r3:
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
        st.markdown(card("Difference", f"{arrow} {abs(delta)*100:.1f}", " pts",
                         "change produced by the modified inputs",
                         "#ef4444" if delta > 0 else ("#34d399" if delta < 0 else MUTED)),
                    unsafe_allow_html=True)

    changed = [f for f in du.FEATURES if abs(what_if[f] - values[f]) > 1e-9]
    fig = go.Figure(go.Bar(x=["Baseline", "Scenario"], y=[score * 100, wf["score"] * 100],
                           marker_color=[colour, d_col],
                           text=[f"{score*100:.1f}%", f"{wf['score']*100:.1f}%"],
                           textposition="outside", width=[0.42, 0.42]))
    fig.add_hline(y=cfg.DECISION_THRESHOLD * 100, line_dash="dot", line_color=MUTED,
                  annotation_text=f"decision threshold {cfg.DECISION_THRESHOLD*100:.0f}%",
                  annotation_font_color=MUTED, annotation_font_size=10)
    fig.update_layout(**PLOT_LAYOUT, height=290,
                      yaxis=dict(range=[0, 108], title="ANN score (%)",
                                 gridcolor=BORDER, zerolinecolor=BORDER),
                      xaxis=dict(showgrid=False),
                      title=dict(text="Changed: " + (", ".join(du.FEATURE_META[f]["short"]
                                 for f in changed) if changed else "nothing yet"),
                                 font=dict(size=12, color=MUTED)))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ==================== SECTION 4 — RECENT MAY DATA ====================
    section(f"Recent {cfg.TARGET_MONTH} data  —  observations vs ANN score")
    years = sorted({d[:4] for d in DF["date"]})
    pick = st.selectbox("Year", years, index=len(years) - 1, key="hist_year")
    sub = DF[DF["date"].str.startswith(pick)].reset_index(drop=True)
    if len(sub) == 0:
        st.info("No records for that year.")
    else:
        scores = NET.predict_proba(SCALER.transform(
            sub[du.FEATURES].to_numpy(float))).ravel()
        days = [int(d[-2:]) for d in sub["date"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=sub["max_temp"], name="Max temperature (°C)",
                                 mode="lines+markers", line=dict(color="#f59e0b", width=2),
                                 marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=days, y=scores * 100, name="ANN heat-wave score (%)",
                                 mode="lines", yaxis="y2",
                                 line=dict(color="#ef4444", width=2, dash="dot")))
        hot = sub.index[sub[du.TARGET] == 1]
        if len(hot):
            fig.add_trace(go.Scatter(x=[days[i] for i in hot],
                                     y=[sub["max_temp"][i] for i in hot], mode="markers",
                                     name="Labelled heat-wave day",
                                     marker=dict(color="#ef4444", size=11,
                                                 symbol="circle-open", line=dict(width=2))))
        fig.update_layout(**PLOT_LAYOUT, height=340,
                          xaxis=dict(title=f"Day of {cfg.TARGET_MONTH}",
                                     gridcolor=BORDER, dtick=2),
                          yaxis=dict(title="°C", gridcolor=BORDER),
                          yaxis2=dict(title="ANN score (%)", overlaying="y", side="right",
                                      range=[0, 100], showgrid=False),
                          legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ==================== SECTION 5 — MODEL STATUS =======================
    model_status_block()


def model_status_block():
    section("Model status")
    s = st.columns(4)
    with s[0]:
        st.markdown(card("ANN architecture",
                         f"{NET.n_input} → {NET.n_hidden} → {NET.n_output}", "",
                         "one hidden layer, implemented in NumPy"), unsafe_allow_html=True)
    with s[1]:
        st.markdown(card("Activation", "Sigmoid", "",
                         "hidden and output layers · BCE loss"), unsafe_allow_html=True)
    with s[2]:
        st.markdown(card("Training",
                         "Completed" if MODEL_READY else "Not trained", "",
                         f"{cfg.TRAIN_EPOCHS} epochs · η = {cfg.TRAIN_LR} · best "
                         f"validation epoch {getattr(NET, 'best_epoch', cfg.TRAIN_EPOCHS)}",
                         "#34d399" if MODEL_READY else "#f59e0b"), unsafe_allow_html=True)
    with s[3]:
        st.markdown(card("Dataset", "DEMO" if META["is_demo"] else "Actual", "",
                         f"{META['rows']} records · {META['n_heatwave']} heat-wave days",
                         "#f59e0b" if META["is_demo"] else "#34d399"),
                    unsafe_allow_html=True)


# ==========================================================================
# 8. PAGE 2 — NEURAL LABORATORY
# ==========================================================================
def init_sim(from_trained: bool):
    """The laboratory keeps its OWN copy so the deployed model can never be damaged."""
    st.session_state.sim_net = NET.copy() if from_trained else A.SimpleANN(
        len(du.FEATURES), cfg.N_HIDDEN, 1,
        seed=int(np.random.default_rng().integers(1, 9999)))
    st.session_state.sim_from_trained = from_trained
    st.session_state.sim_steps = 0
    st.session_state.sim_epochs = 0
    st.session_state.sim_losses = []
    st.session_state.sim_epoch_losses = []
    st.session_state.sim_last = None
    st.session_state.sim_forward = None
    st.session_state.sim_forward_at = 0
    st.session_state.sim_sample = 0


def page_ann_sim():
    st.markdown('<div class="hg-brand" style="font-size:1.5rem">NEURAL LABORATORY</div>'
                '<div class="hg-sub">See how the ANN actually learns — every number below '
                'is computed by <code>ann.py</code> at the moment you press the button.'
                '</div>', unsafe_allow_html=True)

    if "sim_net" not in st.session_state:
        init_sim(from_trained=False)
    sim: A.SimpleANN = st.session_state.sim_net

    st.write("")
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    with c1:
        mode = st.radio("Laboratory network", ["Fresh random weights", "Trained weights"],
                        index=1 if st.session_state.sim_from_trained else 0,
                        horizontal=True, key="sim_mode_radio",
                        help="Start from untrained weights to watch learning happen, or "
                             "from the trained weights to inspect the deployed model.")
        if (mode == "Trained weights") != st.session_state.sim_from_trained:
            init_sim(mode == "Trained weights")
            st.rerun()
    with c2:
        lr = st.number_input("Learning rate η", 0.01, 5.0, cfg.TRAIN_LR, 0.05, key="sim_lr")
    with c3:
        st.markdown(card("Steps / epochs",
                         f"{st.session_state.sim_steps} / {st.session_state.sim_epochs}",
                         "", "single-sample updates / full epochs"), unsafe_allow_html=True)
    with c4:
        st.write("")
        if st.button("↺  Reset Simulation", width="stretch", key="btn_reset_sim"):
            init_sim(st.session_state.sim_from_trained)
            st.rerun()

    # ------------------------------------------------------ architecture --
    section(f"Architecture  —  {sim.n_input} inputs → {sim.n_hidden} hidden → "
            f"{sim.n_output} output")
    left, right = st.columns([2.3, 1])

    with right:
        choices = [f"H{j+1}" for j in range(sim.n_hidden)] + ["Output"]
        neuron = st.selectbox("Select neuron", choices, key="sim_neuron")
        is_output = neuron == "Output"
        j = None if is_output else int(neuron[1:]) - 1
        focus = st.checkbox("Show only this neuron's connections",
                            value=False, key="sim_focus", disabled=is_output)
        if is_output:
            st.markdown('<div class="hg-note">The output neuron\'s eight weights are read '
                        'directly from <code>sim_net.W2[:, 0]</code>.</div>',
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "hidden → output": [f"H{k+1}" for k in range(sim.n_hidden)],
                "weight": [f"{w:+.4f}" for w in sim.W2[:, 0]]}),
                width="stretch", hide_index=True)
            mono(f"output bias b = {float(sim.b2[0, 0]):+.4f}")
        else:
            st.markdown(f'<div class="hg-note">Weights below are read directly from '
                        f'<code>sim_net.W1[:, {j}]</code> and '
                        f'<code>sim_net.W2[{j}, 0]</code>. Orange = positive '
                        f'(excitatory), blue = negative (inhibitory).</div>',
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                f"input → H{j+1}": [du.FEATURE_META[f]["label"] for f in du.FEATURES],
                "weight": [f"{w:+.4f}" for w in sim.W1[:, j]]}),
                width="stretch", hide_index=True)
            mono(f"bias b_{j+1}        = {float(sim.b1[0, j]):+.4f}\n"
                 f"H{j+1} → output w = {float(sim.W2[j, 0]):+.4f}\n"
                 f"output bias b   = {float(sim.b2[0, 0]):+.4f}")

    with left:
        st.plotly_chart(network_figure(sim, activations=st.session_state.sim_forward,
                                       stage=4,
                                       selected=j if (focus and not is_output) else None),
                        width="stretch", config={"displayModeBar": False})

    with st.expander(f"Full input → hidden weight matrix W1 "
                     f"({sim.n_input} × {sim.n_hidden})"):
        wdf = pd.DataFrame(np.round(sim.W1, 4),
                           index=[du.FEATURE_META[f]["label"] for f in du.FEATURES],
                           columns=[f"H{k+1}" for k in range(sim.n_hidden)])
        fig = go.Figure(go.Heatmap(z=sim.W1, x=list(wdf.columns), y=list(wdf.index),
                                   colorscale=[[0, ACCENT], [0.5, "#0d141d"], [1, "#f59e0b"]],
                                   zmid=0, text=np.round(sim.W1, 2), texttemplate="%{text}",
                                   textfont=dict(size=10),
                                   colorbar=dict(title="w", thickness=12)))
        fig.update_layout(**PLOT_LAYOUT, height=300)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.dataframe(wdf, width="stretch")

    # ================================ FORWARD PROPAGATION ================
    section("▶  Forward propagation")
    fp1, fp2 = st.columns([1, 2])
    with fp1:
        n_rows = len(PREP["X_train"])
        idx = int(st.number_input("Training example #", 0, max(n_rows - 1, 0),
                                  int(st.session_state.sim_sample), 1,
                                  key="sim_sample_pick"))
        st.session_state.sim_sample = idx
        run_fwd = st.button("▶  Run Forward Pass", type="primary",
                            width="stretch", key="btn_forward")
    with fp2:
        row = PREP["idx"]["train"][idx]
        raw_row = DF.iloc[row]
        mono(f'date        {raw_row["date"]}   (training row {row})\n'
             + "\n".join(f'{du.FEATURE_META[f]["label"]:<22}{raw_row[f]:>8.1f} '
                         f'{du.FEATURE_META[f]["unit"]:<5} → scaled '
                         f'{PREP["X_train"][idx][i]:.4f}'
                         for i, f in enumerate(du.FEATURES))
             + f'\ntrue heatwave_label   {int(raw_row[du.TARGET])}')

    x_vec = PREP["X_train"][idx]
    y_true = float(PREP["y_train"][idx][0])
    total_updates = st.session_state.sim_steps + st.session_state.sim_epochs

    if run_fwd:
        cache = sim.forward(x_vec)
        st.session_state.sim_forward = {"x": x_vec, "a1": cache["a1"][0],
                                        "a2": float(cache["a2"][0, 0]),
                                        "z1": cache["z1"][0],
                                        "z2": float(cache["z2"][0, 0])}
        st.session_state.sim_forward_at = total_updates
        # Restrained animation: the signal flows input -> hidden -> output.
        holder = st.empty()
        for stage in (0, 1, 2, 3, 4):
            with holder.container():
                st.plotly_chart(network_figure(sim, st.session_state.sim_forward,
                                               stage=stage, height=380),
                                width="stretch", config={"displayModeBar": False},
                                key=f"anim_{stage}")
            time.sleep(0.32)
        holder.empty()

    fwd = st.session_state.sim_forward
    if fwd is None:
        st.info("Press **Run Forward Pass** to push this example through the network.")
    else:
        if st.session_state.sim_forward_at != total_updates:
            st.warning("The weights have changed since this forward pass was run. "
                       "Press **Run Forward Pass** again to refresh these numbers.",
                       icon="⚠️")

        st.latex(r"\text{1. weighted sum:}\quad net_j=\sum_{i=1}^{6}x_i w_{ij}+b_j"
                 r"\qquad\text{2. activation:}\quad h_j=\sigma(net_j)="
                 r"\frac{1}{1+e^{-net_j}}")

        # ---- EXPLAIN ONE NEURON: input x weight = contribution -----------
        st.markdown(f"**Explain one neuron — {neuron}**")
        if is_output:
            terms = fwd["a1"] * sim.W2[:, 0]
            st.dataframe(pd.DataFrame({
                "input to this neuron": [f"H{k+1} activation" for k in range(sim.n_hidden)],
                "value": np.round(fwd["a1"], 5),
                "× weight": np.round(sim.W2[:, 0], 5),
                "= contribution": np.round(terms, 5)}), width="stretch", hide_index=True)
            net_val, act = fwd["z2"], fwd["a2"]
            bias = float(sim.b2[0, 0])
        else:
            terms = fwd["x"] * sim.W1[:, j]
            st.dataframe(pd.DataFrame({
                "input": [du.FEATURE_META[f]["label"] for f in du.FEATURES],
                "raw value": [f'{raw_row[f]:.1f} {du.FEATURE_META[f]["unit"]}'
                              for f in du.FEATURES],
                "scaled value x": np.round(fwd["x"], 5),
                "× weight w": np.round(sim.W1[:, j], 5),
                "= contribution x·w": np.round(terms, 5)}),
                width="stretch", hide_index=True)
            net_val, act = float(fwd["z1"][j]), float(fwd["a1"][j])
            bias = float(sim.b1[0, j])
        st.caption("The network consumes **min–max scaled** inputs, so the arithmetic "
                   "uses the scaled value. The raw value is shown alongside for context.")
        mono(f"Σ contributions = {terms.sum():+.6f}\n"
             f"+ bias          = {bias:+.6f}\n"
             f"net             = {net_val:+.6f}\n"
             f"activation      = sigmoid(net) = {act:.6f}")

        st.markdown("**All hidden neurons, then the output neuron**")
        st.dataframe(pd.DataFrame({
            "neuron": [f"H{k+1}" for k in range(sim.n_hidden)],
            "net_j": np.round(fwd["z1"], 5),
            "h_j = σ(net_j)": np.round(fwd["a1"], 5),
            "w_j → output": np.round(sim.W2[:, 0], 5),
            "h_j · w_j": np.round(fwd["a1"] * sim.W2[:, 0], 5)}),
            width="stretch", hide_index=True)

        st.latex(r"\hat{y}=\sigma\!\left(\sum_{j=1}^{8}h_j w_j + b\right)")
        verdict = "HEAT WAVE" if fwd["a2"] >= cfg.DECISION_THRESHOLD else "NO HEAT WAVE"
        ok = (fwd["a2"] >= cfg.DECISION_THRESHOLD) == (y_true >= 0.5)
        mono(f'Σ h_j·w_j        = {float(np.sum(fwd["a1"] * sim.W2[:, 0])):+.6f}\n'
             f'+ output bias b  = {float(sim.b2[0, 0]):+.6f}\n'
             f'z                = {fwd["z2"]:+.6f}\n'
             f'ŷ = sigmoid(z)   = {fwd["a2"]:.6f}   →   {fwd["a2"]*100:.2f} %\n'
             f'true label y     = {y_true:.0f}\n'
             f'prediction       = {verdict} ({"correct" if ok else "wrong"})')

    # ==================================== LEARNING =======================
    section("▶  Learning  —  backpropagation + weight update")
    st.markdown('<div class="hg-note"><b>Learn One Step</b> runs '
                '<code>SimpleANN.learn_one_step()</code>: forward pass → error → '
                'backpropagation → gradient-descent update, on the one training example '
                'selected above. <b>Train 10 Epochs</b> runs the same three operations '
                'over the whole training set, ten times. Start from <b>fresh random '
                'weights</b> to see the error actually fall.</div>',
                unsafe_allow_html=True)
    st.write("")

    l1, l2, l3, _ = st.columns([1, 1, 1, 1.4])
    with l1:
        step_once = st.button("▶  Learn One Step", type="primary",
                              width="stretch", key="btn_step1")
    with l2:
        step_many = st.button("⏩  Run 50 Steps", width="stretch", key="btn_step50")
    with l3:
        train10 = st.button("⏱  Train 10 Epochs", width="stretch", key="btn_train10")

    if step_once or step_many:
        n = 50 if step_many else 1
        rec = None
        for k in range(n):
            pick = idx if n == 1 else int(np.random.default_rng(
                st.session_state.sim_steps + k).integers(0, len(PREP["X_train"])))
            rec = sim.learn_one_step(PREP["X_train"][pick],
                                     float(PREP["y_train"][pick][0]),
                                     learning_rate=lr,
                                     watch_hidden=(0 if j is None else j))
            st.session_state.sim_steps += 1
            st.session_state.sim_losses.append(rec["loss_after"])
        st.session_state.sim_last = rec
        # Redraw the whole page so the weight tables, the diagram and the
        # forward-pass panel all reflect the SAME (new) weights. Without this
        # the forward-pass section above would still show the pre-update values.
        st.rerun()

    if train10:
        losses = sim.train_epochs(PREP["X_train"], PREP["y_train"],
                                  n_epochs=10, learning_rate=lr)
        st.session_state.sim_epoch_losses += losses
        st.session_state.sim_epochs += 10
        st.rerun()

    rec = st.session_state.sim_last
    if rec is None and not st.session_state.sim_epoch_losses:
        st.info("Press **Learn One Step** or **Train 10 Epochs** to update the weights.")

    if rec is not None:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(card("Step", st.session_state.sim_steps, "",
                             "single-sample updates"), unsafe_allow_html=True)
        with m2:
            st.markdown(card("Target y", f"{rec['target']:.0f}", "", "1 = heat wave"),
                        unsafe_allow_html=True)
        with m3:
            st.markdown(card("Prediction ŷ", f"{rec['pred_before']:.4f}", "",
                             f"after update → {rec['pred_after']:.4f}"),
                        unsafe_allow_html=True)
        with m4:
            better = rec["loss_after"] < rec["loss_before"]
            st.markdown(card("Loss E", f"{rec['loss_before']:.4f}", "",
                             f"after update → {rec['loss_after']:.4f} "
                             f"({'lower ✓' if better else 'higher'})",
                             "#34d399" if better else "#f59e0b"), unsafe_allow_html=True)

        st.latex(r"E=-\big[y\log\hat{y}+(1-y)\log(1-\hat{y})\big]"
                 r"\qquad \delta_{out}=\frac{\partial E}{\partial z}=\hat{y}-y"
                 r"\qquad w_{new}=w_{old}-\eta\frac{\partial E}{\partial w}")
        mono(f'raw error (y − ŷ)    = {rec["raw_error"]:+.6f}\n'
             f'output delta (ŷ − y) = {rec["output_delta"]:+.6f}\n'
             f'learning rate η      = {rec["learning_rate"]}')

        st.markdown(f"**Weight update — the six weights feeding hidden neuron "
                    f"H{rec['watch_hidden']+1}**")
        st.dataframe(pd.DataFrame({
            "weight": [f"w[{du.FEATURE_META[f]['short']} → H{rec['watch_hidden']+1}]"
                       for f in du.FEATURES],
            "old weight": np.round(rec["w1_old"], 6),
            "gradient ∂E/∂w": np.round(rec["w1_grad"], 6),
            "ΔW = −η·grad": np.round(rec["w1_delta"], 6),
            "new weight": np.round(rec["w1_new"], 6)}), width="stretch", hide_index=True)

        st.markdown("**Weight update — the eight hidden → output weights**")
        st.dataframe(pd.DataFrame({
            "weight": [f"w[H{k+1} → out]" for k in range(sim.n_hidden)],
            "old weight": np.round(rec["w2_old"], 6),
            "gradient ∂E/∂w": np.round(rec["w2_grad"], 6),
            "ΔW = −η·grad": np.round(rec["w2_delta"], 6),
            "new weight": np.round(rec["w2_new"], 6)}), width="stretch", hide_index=True)
        mono(f'output bias  b: {rec["b2_old"]:+.6f}  →  {rec["b2_new"]:+.6f}')

    # ---------------------------- loss curves for the laboratory ---------
    lc1, lc2 = st.columns(2)
    with lc1:
        if len(st.session_state.sim_losses) > 1:
            fig = go.Figure(go.Scatter(y=st.session_state.sim_losses, mode="lines",
                                       line=dict(color=ACCENT, width=2)))
            fig.update_layout(**PLOT_LAYOUT, height=280,
                              xaxis=dict(title="learning step", gridcolor=BORDER),
                              yaxis=dict(title="binary cross-entropy E", gridcolor=BORDER),
                              title=dict(text="Error after each single-sample step",
                                         font=dict(size=12, color=MUTED)))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Single-sample (stochastic) updates are noisy by nature — the "
                       "trend matters, not each spike.")
    with lc2:
        if len(st.session_state.sim_epoch_losses) > 1:
            fig = go.Figure(go.Scatter(y=st.session_state.sim_epoch_losses, mode="lines",
                                       line=dict(color="#f59e0b", width=2)))
            fig.update_layout(**PLOT_LAYOUT, height=280,
                              xaxis=dict(title="epoch", gridcolor=BORDER),
                              yaxis=dict(title="training loss", gridcolor=BORDER),
                              title=dict(text="Loss vs epoch (full-batch training)",
                                         font=dict(size=12, color=MUTED)))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ============================ WEIGHT LEARNING VISUALISATION ==========
    section("Weight learning  —  initial weights vs current weights")
    pairs, initial, current = sim.weight_change_table()
    change = current - initial
    names = [f"{du.FEATURE_META[du.FEATURES[i]]['short']} → H{jj+1}" for i, jj in pairs]
    table = pd.DataFrame({"connection": names,
                          "initial": np.round(initial, 5),
                          "current": np.round(current, 5),
                          "change": np.round(change, 5)})
    if float(np.abs(change).max()) < 1e-12:
        st.info("This network has not been updated yet — initial and current weights are "
                "identical. Press **Learn One Step** or **Train 10 Epochs**.")
    top = table.reindex(table["change"].abs().sort_values(ascending=False).index).head(10)
    tv1, tv2 = st.columns([1.15, 1])
    with tv1:
        st.markdown("**Ten connections that moved the most**")
        st.dataframe(top, width="stretch", hide_index=True)
    with tv2:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top["connection"][::-1], x=top["change"][::-1],
                             orientation="h",
                             marker_color=["#f59e0b" if c >= 0 else ACCENT
                                           for c in top["change"][::-1]]))
        fig.update_layout(**PLOT_LAYOUT, height=340,
                          xaxis=dict(title="change (current − initial)", gridcolor=BORDER),
                          yaxis=dict(gridcolor=BORDER),
                          title=dict(text="How far each weight has learned",
                                     font=dict(size=12, color=MUTED)))
        fig.update_layout(margin=dict(l=110, r=20, t=45, b=40))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with st.expander(f"All {len(table)} input → hidden connections"):
        st.dataframe(table, width="stretch", hide_index=True, height=320)


# ==========================================================================
# 9. PAGE 3 — MODEL
# ==========================================================================
def page_model():
    st.markdown('<div class="hg-brand" style="font-size:1.5rem">MODEL &amp; DATA</div>'
                '<div class="hg-sub">Dataset explorer, training curves and test-set '
                'evaluation.</div>', unsafe_allow_html=True)

    section("Data explorer")
    k = st.columns(4)
    with k[0]:
        st.markdown(card("Records", META["rows"], "",
                         f"{cfg.TARGET_MONTH} days in the dataset"), unsafe_allow_html=True)
    with k[1]:
        st.markdown(card("Heat-wave days", META["n_heatwave"], "",
                         f"{META['n_heatwave']/META['rows']*100:.1f}% of records",
                         "#ef4444"), unsafe_allow_html=True)
    with k[2]:
        st.markdown(card("Normal days", META["n_normal"], "",
                         f"{META['n_normal']/META['rows']*100:.1f}% of records",
                         "#34d399"), unsafe_allow_html=True)
    with k[3]:
        st.markdown(card(f"{cfg.TARGET_MONTH} normal Tmax",
                         f"{META['normal_max_temp']:.2f}", " °C",
                         "mean of this dataset (not an IMD published normal)"),
                    unsafe_allow_html=True)

    with st.expander("Basic statistics"):
        st.dataframe(DF[du.FEATURES].describe().T.round(2), width="stretch")
    with st.expander("Raw records"):
        st.dataframe(DF, width="stretch", height=320)

    st.write("")
    t1, t2, t3 = st.tabs(["Maximum temperature", "Humidity", "Heat-wave labels"])
    with t1:
        fig = go.Figure(go.Scatter(x=DF["date"], y=DF["max_temp"], mode="lines",
                                   line=dict(color="#f59e0b", width=1.3)))
        fig.add_hline(y=META["normal_max_temp"], line_dash="dot", line_color=MUTED,
                      annotation_text=f"dataset {cfg.TARGET_MONTH} mean",
                      annotation_font_color=MUTED, annotation_font_size=10)
        fig.update_layout(**PLOT_LAYOUT, height=320,
                          yaxis=dict(title="°C", gridcolor=BORDER),
                          xaxis=dict(gridcolor=BORDER, title="date"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with t2:
        fig = go.Figure(go.Scatter(x=DF["date"], y=DF["humidity"], mode="lines",
                                   line=dict(color=ACCENT, width=1.3)))
        fig.update_layout(**PLOT_LAYOUT, height=320,
                          yaxis=dict(title="%", gridcolor=BORDER),
                          xaxis=dict(gridcolor=BORDER, title="date"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with t3:
        fig = go.Figure(go.Bar(x=DF["date"], y=DF[du.TARGET],
                               marker_color=["#ef4444" if v else BORDER
                                             for v in DF[du.TARGET]]))
        fig.update_layout(**PLOT_LAYOUT, height=320,
                          yaxis=dict(title="heatwave_label", tickvals=[0, 1],
                                     gridcolor=BORDER),
                          xaxis=dict(gridcolor=BORDER, title="date"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    section("How the label was produced")
    st.markdown(hw.RULE_TEXT)
    st.caption(f"{cfg.TARGET_MONTH} normal used for Condition A: "
               f"{META['normal_max_temp']:.2f} °C (the mean of the loaded dataset). "
               "Conditions A and B follow IMD's coastal criteria; Condition C is this "
               "project's own humid-heat rule and is not an official meteorological "
               "definition.")

    section("Training configuration")
    s = st.columns(4)
    for col, (lab, key) in zip(s, [("Training samples", "train"),
                                   ("Validation samples", "val"),
                                   ("Testing samples", "test")]):
        n = len(PREP[f"X_{key}"])
        pos = int(PREP[f"y_{key}"].sum())
        with col:
            st.markdown(card(lab, n, "", f"{pos} heat-wave / {n - pos} normal"),
                        unsafe_allow_html=True)
    with s[3]:
        st.markdown(card("Architecture",
                         f"{NET.n_input} → {NET.n_hidden} → {NET.n_output}", "",
                         f"sigmoid · BCE · η = {cfg.TRAIN_LR} · {cfg.TRAIN_EPOCHS} epochs"),
                    unsafe_allow_html=True)
    st.caption("The split is stratified and random (seed "
               f"{cfg.SEED}). The min–max scaler is fitted on the training rows only, so "
               "no information from the validation or test rows reaches the model. The "
               "weights kept are those with the lowest validation loss (epoch "
               f"{getattr(NET, 'best_epoch', cfg.TRAIN_EPOCHS)}).")

    with st.expander("Normalisation parameters (fitted on training data only)"):
        st.dataframe(SCALER.as_frame(), width="stretch", hide_index=True)

    section("Training curves")
    h = NET.history
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["train_loss"], name="training loss",
                                 line=dict(color=ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_loss"], name="validation loss",
                                 line=dict(color="#f59e0b", width=2, dash="dot")))
        fig.update_layout(**PLOT_LAYOUT, height=320,
                          xaxis=dict(title="epoch", gridcolor=BORDER),
                          yaxis=dict(title="binary cross-entropy", gridcolor=BORDER),
                          legend=dict(orientation="h", y=1.15, font=dict(size=10)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with cc2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["epoch"], y=np.array(h["train_acc"]) * 100,
                                 name="training accuracy", line=dict(color=ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=h["epoch"], y=np.array(h["val_acc"]) * 100,
                                 name="validation accuracy",
                                 line=dict(color="#f59e0b", width=2, dash="dot")))
        fig.update_layout(**PLOT_LAYOUT, height=320,
                          xaxis=dict(title="epoch", gridcolor=BORDER),
                          yaxis=dict(title="accuracy (%)", gridcolor=BORDER, range=[0, 102]),
                          legend=dict(orientation="h", y=1.15, font=dict(size=10)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    section("Evaluation on the held-out test set")
    m = METRICS["test"]
    e = st.columns(4)
    for col, (name, val) in zip(e, [("Accuracy", m["accuracy"]),
                                    ("Precision", m["precision"]),
                                    ("Recall", m["recall"]), ("F1 score", m["f1"])]):
        with col:
            st.markdown(card(name, f"{val*100:.1f}", " %",
                             "computed from real test predictions"), unsafe_allow_html=True)

    cm1, cm2 = st.columns(2)
    with cm1:
        z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        fig = go.Figure(go.Heatmap(z=z, x=["predicted: no heat wave", "predicted: heat wave"],
                                   y=["actual: no heat wave", "actual: heat wave"],
                                   colorscale=[[0, "#0d141d"], [1, ACCENT]], showscale=False,
                                   text=z, texttemplate="%{text}",
                                   textfont=dict(size=22, color=TEXT)))
        fig.update_layout(**PLOT_LAYOUT, height=300,
                          title=dict(text="Confusion matrix (test set)",
                                     font=dict(size=12, color=MUTED)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with cm2:
        rows = [{"split": s, "n": METRICS[s]["n"],
                 "heat-wave rows": METRICS[s]["n_positive"],
                 "accuracy": f"{METRICS[s]['accuracy']*100:.1f}%",
                 "precision": f"{METRICS[s]['precision']*100:.1f}%",
                 "recall": f"{METRICS[s]['recall']*100:.1f}%",
                 "F1": f"{METRICS[s]['f1']*100:.1f}%"} for s in ("train", "val", "test")]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown(
            f'<div class="hg-note"><b>Read these numbers carefully.</b><br>'
            f'• The test set holds only <b>{m["n"]} rows, {m["n_positive"]} of them '
            f'heat-wave days</b>. With so few positives, precision and recall move by '
            f'large steps and are not statistically reliable.<br>'
            f'• <b>The target label is a deterministic function of two of the six '
            f'inputs</b> (max temperature and humidity, via the rule above). A high score '
            f'therefore means "the ANN successfully learned the labelling rule", '
            f'<b>not</b> "the ANN can forecast real heat waves".<br>'
            f'{"• The data is SYNTHETIC DEMO DATA, so these metrics say nothing about real-world skill.<br>" if META["is_demo"] else ""}'
            f'</div>', unsafe_allow_html=True)


# ==========================================================================
# 10. PAGE 4 — METHODOLOGY
# ==========================================================================
def page_methodology():
    st.markdown('<div class="hg-brand" style="font-size:1.5rem">METHODOLOGY</div>'
                '<div class="hg-sub">Everything needed to explain this project in a '
                'viva.</div>', unsafe_allow_html=True)

    section("Project flow")
    fl, fr = st.columns([1, 1.25])
    with fl:
        st.plotly_chart(flow_figure(), width="stretch", config={"displayModeBar": False})
    with fr:
        section("Problem")
        st.markdown(
            f"Given six routinely-measured climate variables for a {cfg.TARGET_MONTH} day "
            f"in {cfg.STUDY_AREA['name']}, decide whether **heat-wave conditions are "
            "likely** for that day. A supervised **binary classification** problem, "
            "solved with an ANN implemented from scratch in NumPy so that every Soft "
            "Computing step stays visible.")

        section("Input data")
        st.dataframe(pd.DataFrame([
            {"#": i + 1, "variable": f, "meaning": du.FEATURE_META[f]["label"],
             "unit": du.FEATURE_META[f]["unit"]}
            for i, f in enumerate(du.FEATURES)]), width="stretch", hide_index=True)
        st.markdown("**Target:** `heatwave_label` — `1` = heat wave, `0` = no heat wave, "
                    "produced by the documented project rule:")
        st.markdown(hw.RULE_TEXT)
        st.info("**Condition C is this project's own rule**, built on the NOAA heat-index "
                "formula. Conditions A and B mirror IMD's criteria for a coastal station. "
                "The project rule and the official meteorological definition are kept "
                "deliberately distinct.", icon="📌")

        section("Preprocessing")
        st.markdown("The six inputs live on very different scales, so each is **min–max "
                    "scaled** into roughly [0, 1]:")
        st.latex(r"x_{scaled}=\frac{x-x_{min}}{x_{max}-x_{min}}")
        st.markdown("`x_min` / `x_max` come from the **training rows only** and are reused "
                    "unchanged for validation, test and every live prediction. Fitting "
                    "them on all the data would leak test-set information.")
        st.dataframe(SCALER.as_frame(), width="stretch", hide_index=True)

    section("ANN architecture")
    st.markdown(f"**{NET.n_input} input neurons → {NET.n_hidden} hidden neurons (sigmoid) "
                f"→ {NET.n_output} output neuron (sigmoid)**  —  one hidden layer makes "
                "the network a universal approximator while staying small enough to read "
                "off completely.")
    st.plotly_chart(network_figure(NET, height=380), width="stretch",
                    config={"displayModeBar": False})

    section("Activation function")
    st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}\qquad\qquad "
             r"\sigma'(z)=\sigma(z)\big(1-\sigma(z)\big)")
    st.markdown("Smooth, differentiable everywhere, and it squashes any real number into "
                "(0, 1). Without a non-linear activation, stacking layers would collapse "
                "into a single linear function and no curved decision boundary could be "
                "learned.")

    section("Forward propagation")
    st.latex(r"net_j=\sum_{i=1}^{6}x_i\,w_{ij}+b_j \qquad h_j=\sigma(net_j) "
             r"\qquad \hat{y}=\sigma\!\left(\sum_{j=1}^{8}h_j w_j+b\right)")
    st.markdown("The **weight** `w_ij` says how strongly input *i* matters to neuron *j*; "
                "the **bias** `b_j` shifts the neuron's threshold so it can fire even when "
                "every input is zero.")

    section("Error")
    st.latex(r"E=-\frac{1}{m}\sum_{k=1}^{m}\Big[y_k\log\hat{y}_k+"
             r"(1-y_k)\log(1-\hat{y}_k)\Big]")
    st.markdown("Binary cross-entropy: the natural loss for a 0/1 target with a sigmoid "
                "output. It punishes a confident wrong answer far more heavily than MSE.")

    section("Backpropagation")
    st.latex(r"\delta^{out}=\frac{\partial E}{\partial z}=\hat{y}-y")
    st.latex(r"\frac{\partial E}{\partial w_j}=\delta^{out}h_j \qquad\qquad "
             r"\delta^{hid}_j=\delta^{out}w_j\,h_j(1-h_j) \qquad\qquad "
             r"\frac{\partial E}{\partial w_{ij}}=\delta^{hid}_j\,x_i")
    st.markdown("The chain rule run backwards. With a sigmoid output *and* cross-entropy "
                "loss the two derivatives cancel, leaving the clean `ŷ − y`.")

    section("Weight update")
    st.latex(r"w_{new}=w_{old}-\eta\,\frac{\partial E}{\partial w}\qquad "
             r"b_{new}=b_{old}-\eta\,\frac{\partial E}{\partial b}")
    st.markdown(f"η (learning rate) = **{cfg.TRAIN_LR}**. Too small and learning crawls; "
                "too large and the updates overshoot and oscillate.")

    section("Training process")
    st.markdown(f"""
1. Split the {META['rows']} records into training / validation / test
   ({len(PREP['X_train'])} / {len(PREP['X_val'])} / {len(PREP['X_test'])}), stratified by label.
2. Fit the min–max scaler on the **training rows only**; apply it everywhere.
3. Initialise the weights to small random values (Xavier range), biases to zero.
4. For each of **{cfg.TRAIN_EPOCHS} epochs**: forward pass → binary cross-entropy →
   backpropagation → gradient-descent update of every weight and bias.
5. Track the validation loss and keep the weights from the best epoch
   (epoch **{getattr(NET, 'best_epoch', cfg.TRAIN_EPOCHS)}**) — simple early stopping,
   the project's defence against overfitting.
6. Evaluate once on the untouched test set.
""")

    section("Prediction")
    st.markdown(
        "A new day's six values are scaled with the stored training statistics, pushed "
        "through one forward pass, and the sigmoid output is reported as the **ANN "
        f"estimated heat-wave score**. A score ≥ {cfg.DECISION_THRESHOLD:.2f} is reported "
        "as HEAT WAVE. The risk wording "
        f"(LOW < {cfg.RISK_LOW_MAX:.2f} ≤ MODERATE < {cfg.RISK_MODERATE_MAX:.2f} ≤ HIGH) "
        "consists of **project risk display thresholds** kept in `config.py`. They are a "
        "presentation choice for this demonstration and are **not** official "
        "meteorological thresholds.")

    section("Limitations  —  state these honestly in the viva")
    st.markdown(f"""
- {"**The bundled dataset is SYNTHETIC DEMO DATA**, generated for application testing. No conclusion about real Mumbai weather can be drawn from it." if META['is_demo'] else "The dataset supplied is treated as genuine; its provenance is the user's responsibility."}
- The label is produced by a **deterministic rule** applied to max temperature and
  humidity, so the ANN can learn it very well. High accuracy demonstrates that the
  learning algorithm works, **not** that the system can forecast real heat waves.
- The sigmoid output is **not a calibrated probability** — no calibration (Platt scaling,
  isotonic regression) has been performed, so it is always labelled an *estimated score*.
- The "{cfg.TARGET_MONTH} normal" in Condition A is the mean of the loaded dataset, not an
  IMD-published 30-year climatological normal.
- The test set contains only **{METRICS['test']['n_positive']} heat-wave rows**, so
  precision and recall are statistically fragile.
- The split is random rather than chronological. For a genuine forecasting system,
  training on earlier years and testing on later years would be more honest.
- The model classifies a day from that day's own observations: it is a **diagnostic**
  classifier, not a lead-time forecast.
- **The map marks a single model study area.** One location is modelled, so no
  neighbourhood-level or per-suburb probabilities are produced or displayed. Station-level
  markers appear only if the dataset itself carries real `station` / `latitude` /
  `longitude` columns.
- No formal feature-attribution method (SHAP, permutation importance) is implemented — the
  "Why this prediction?" panel shows the actual arithmetic, nothing more.
""")


# ==========================================================================
# 11. ROUTER
# ==========================================================================
try:
    page = st.session_state.page
    if page == PAGES[0]:
        page_heatwave()
    elif page == PAGES[1]:
        page_ann_sim()
    elif page == PAGES[2]:
        page_model()
    else:
        page_methodology()
except Exception as exc:                                   # friendly, never a traceback
    st.error(f"Something went wrong while drawing this page: "
             f"`{type(exc).__name__}: {exc}`\n\n"
             "Try **Reset Simulation** on the ANN page, or reload the app.")

st.write("")
st.markdown('<div class="hg-note" style="text-align:center; border-top:1px solid #1e2a3a; '
            'padding-top:1rem">HEATGUARD AI · Soft Computing project · ANN written from '
            'scratch in NumPy · Mumbai, May · every model number on this site is computed '
            'at run time</div>', unsafe_allow_html=True)
