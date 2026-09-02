"""ElmanGuard -- Streamlit front end.

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from elmanguard.config import DEFAULT  # noqa: E402
from elmanguard.features import FeaturePipeline  # noqa: E402
from elmanguard.i18n import LANGUAGES, pretty, t  # noqa: E402
from elmanguard.plotting import comparison_figure, timeline_figure  # noqa: E402

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DEMO = ROOT / "data" / "demo_windows.npz"

st.set_page_config(page_title="ElmanGuard", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model():
    from tensorflow import keras

    if not (MODELS / "elman.keras").exists():
        return None, None, None
    artifacts = json.loads((MODELS / "artifacts.json").read_text())
    model = keras.models.load_model(MODELS / "elman.keras")
    pipe = FeaturePipeline.from_dict(artifacts["pipeline"], DEFAULT)
    return model, pipe, artifacts


@st.cache_data(show_spinner=False)
def load_demo():
    if not DEMO.exists():
        return None
    z = np.load(DEMO, allow_pickle=True)
    return {
        "X": z["X"], "y": z["y"], "attack_type": z["attack_type"],
        "host": z["host"], "day": z["day"],
        "end_time": pd.to_datetime(z["end_time"]),
    }


@st.cache_data(show_spinner=False)
def load_reports():
    def read(name, **kw):
        p = REPORTS / name
        return pd.read_csv(p, **kw) if p.exists() else None
    return {
        "pooled": read("pooled_results.csv", index_col=0),
        "within_host": read("within_host.csv"),
        "ablation": read("order_ablation.csv", index_col=0),
        "per_attack": read("per_attack_recall.csv", index_col=0),
        "random": read("random_split/pooled_results.csv", index_col=0),
    }


lang = LANGUAGES[st.sidebar.selectbox("Language / Idioma", list(LANGUAGES), index=1)]
st.title(t(lang, "title"))
st.caption(t(lang, "tagline"))
st.warning(f"**{t(lang, 'disclaimer_title')}** — {t(lang, 'disclaimer')}", icon="🔬")

model, pipe, artifacts = load_model()
demo = load_demo()
reports = load_reports()
dark = st.get_option("theme.base") == "dark"

# --------------------------------------------------------------------------- #
# Live view
# --------------------------------------------------------------------------- #
if model is None or demo is None:
    st.info("Run `python scripts/prepare_data.py`, `scripts/run_experiment.py` and "
            "`scripts/export_demo.py` to populate the demo.")
else:
    st.sidebar.header(t(lang, "sidebar_data"))
    hosts = sorted(set(demo["host"].tolist()))
    host = st.sidebar.selectbox(t(lang, "host_pick"), hosts,
                                index=hosts.index("192.168.10.50") if "192.168.10.50" in hosts else 0)
    st.sidebar.header(t(lang, "sidebar_view"))
    default_thr = float(artifacts["threshold"])
    threshold = st.sidebar.slider(t(lang, "threshold"), 0.0, 1.0, default_thr, 0.01)

    mask = demo["host"] == host
    X = pipe.transform(demo["X"][mask])
    scores = model.predict(X, verbose=0).ravel()
    y = demo["y"][mask]
    times = demo["end_time"][mask]

    flagged = int((scores >= threshold).sum())
    recall = float(((scores >= threshold) & (y == 1)).sum() / max(int(y.sum()), 1))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t(lang, "metric_windows"), f"{mask.sum():,}")
    c2.metric(t(lang, "metric_attack"), f"{int(y.sum()):,}",
              t(lang, "metric_attack_delta", pct=f"{y.mean():.0%}"), delta_color="off")
    c3.metric(t(lang, "metric_flagged"), f"{flagged:,}")
    c4.metric(t(lang, "metric_recall"), f"{recall:.0%}")

    st.subheader(t(lang, "timeline_title"))
    st.caption(t(lang, "timeline_help"))
    st.plotly_chart(
        timeline_figure(times.to_numpy(), scores, y, threshold,
                        {"score": t(lang, "p_attack"),
                         "truth": "ground-truth attack" if lang == "en" else "ataque (rótulo real)",
                         "threshold": "alert threshold" if lang == "en" else "limiar de alerta",
                         "x": "window (in time order)" if lang == "en" else "janela (em ordem temporal)",
                         "y": t(lang, "p_attack")}, dark=dark),
        width="stretch", config={"displayModeBar": False},
    )

    table = pd.DataFrame({
        "time": times, "p_attack": scores, "truth": y,
        "attack_type": demo["attack_type"][mask],
    }).sort_values("time")
    with st.expander("🔢 " + t(lang, "window_table")):
        st.dataframe(table, width="stretch", height=300)
    st.download_button(
        t(lang, "download"), table.to_csv(index=False).encode(),
        file_name=f"elmanguard_{host}.csv", mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# Measured results
# --------------------------------------------------------------------------- #
st.divider()
st.subheader(t(lang, "results_title"))
st.markdown(t(lang, "results_intro"))

if reports["pooled"] is None:
    st.info(t(lang, "no_reports"))
else:
    cols = [c for c in ("pr_auc_mean", "pr_auc_std", "recall_mean", "precision_mean", "f1_mean")
            if c in reports["pooled"].columns]
    st.dataframe(pretty(reports["pooled"][cols], lang), width="stretch")
    st.plotly_chart(
        comparison_figure(reports["pooled"],
                          {"floor": t(lang, "prevalence_floor_label")}, dark=dark, lang=lang),
        width="stretch", config={"displayModeBar": False},
    )
    if reports["per_attack"] is not None:
        st.caption("Recall by attack type — every type below is unseen during training."
                   if lang == "en" else
                   "Recall por tipo de ataque — todos os tipos abaixo são inéditos no treino.")
        st.dataframe(pretty(reports["per_attack"], lang), width="stretch")

    if reports["random"] is not None:
        delta = (reports["random"]["pr_auc_mean"] - reports["pooled"]["pr_auc_mean"]).dropna()
        st.info(
            ("**A random split would report much more.** Same data, same models, "
             "shuffled instead of chronological: PR-AUC rises by "
             if lang == "en" else
             "**Um split aleatório reportaria muito mais.** Mesmos dados, mesmos modelos, "
             "embaralhado em vez de cronológico: o PR-AUC sobe em ")
            + ", ".join(f"{m} +{v:.3f}" for m, v in delta.items() if m != "prevalence_floor")
            + ("." ),
            icon="⚠️",
        )

with st.expander("🎯 " + t(lang, "control_title"), expanded=True):
    st.markdown(t(lang, "control_intro"))
    if reports["within_host"] is not None:
        st.dataframe(pretty(reports["within_host"], lang), width="stretch")

with st.expander("🔀 " + t(lang, "ablation_title"), expanded=True):
    st.markdown(t(lang, "ablation_intro"))
    if reports["ablation"] is not None:
        st.dataframe(pretty(reports["ablation"], lang), width="stretch")

with st.expander("🧠 " + t(lang, "how_title")):
    doc = ROOT / "docs" / f"how_it_works.{lang}.md"
    st.markdown(doc.read_text(encoding="utf-8") if doc.exists() else "See the README.")

st.caption(t(lang, "footer"))
