"""
Tətbiqin ümumi dizayn və məlumat funksiyaları
=============================================

Bütün səhifələr bu fayldan istifadə edir:
  - eyni rəng palitrası və CSS
  - eyni başlıq / izah blokları
  - eyni keşlənmiş (cache) verilənlər və modellər

KEŞ (cache) NƏDİR?
    Streamlit hər düymə basılanda bütün kodu yenidən işlədir.
    @st.cache_data / @st.cache_resource sayəsində ağır hesablamalar
    (CSV oxumaq, model öyrətmək) yalnız BİR DƏFƏ edilir, sonra
    yaddaşdan götürülür. Buna görə tətbiq sürətli işləyir.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import config
from . import evaluation as ev
from .data_loader import film_title_map, load_all
from .models.item_cf import ItemCFRecommender
from .models.popularity import PopularityRecommender
from .models.svd import SVDRecommender
from .split import split_ratings

# --------------------------------------------------------------------------
# PALİTRA
# --------------------------------------------------------------------------

BURGUNDY = config.COLOR_BURGUNDY
GOLD = config.COLOR_GOLD
TEAL = config.COLOR_TEAL
SLATE = config.COLOR_SLATE
INK = config.COLOR_INK
CREAM = config.COLOR_CREAM

# Diaqramlarda ardıcıl istifadə olunan köməkçi rənglər
SEQUENCE = [BURGUNDY, TEAL, GOLD, SLATE, "#A8452F", "#3E7B8C", "#8E6A3F", "#4B5563"]


# --------------------------------------------------------------------------
# CSS VƏ BAŞLIQLAR
# --------------------------------------------------------------------------

def inject_css() -> None:
    """Bütün səhifələr üçün ümumi stil."""
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}

        /* --- Səhifə başlığı (kino lövhəsi görünüşü) --- */
        .hero {{
            background: linear-gradient(120deg, {BURGUNDY} 0%, #5A1530 55%, {TEAL} 160%);
            border-radius: 16px;
            padding: 1.6rem 1.9rem;
            color: #FBF7F1;
            margin-bottom: 1.4rem;
            border-left: 7px solid {GOLD};
            box-shadow: 0 6px 22px rgba(36, 26, 34, 0.18);
        }}
        .hero h1 {{
            margin: 0; font-size: 1.95rem; font-weight: 800;
            letter-spacing: -0.4px; color: #FBF7F1;
        }}
        .hero p {{
            margin: 0.45rem 0 0 0; font-size: 1.02rem;
            color: #EFDFC6; line-height: 1.5;
        }}

        /* --- KPI kartları --- */
        .kpi {{
            background: #FFFFFF;
            border-radius: 13px;
            padding: 1.05rem 1.15rem;
            border-top: 4px solid {GOLD};
            box-shadow: 0 2px 10px rgba(36,26,34,0.09);
            height: 100%;
        }}
        .kpi .label {{
            font-size: 0.79rem; text-transform: uppercase; letter-spacing: 0.8px;
            color: {SLATE}; font-weight: 700; margin-bottom: 0.3rem;
        }}
        .kpi .value {{
            font-size: 1.85rem; font-weight: 800; color: {BURGUNDY};
            line-height: 1.1;
        }}
        .kpi .hint {{ font-size: 0.82rem; color: #6B6169; margin-top: 0.3rem; }}

        /* --- "Bu nə deməkdir?" izah bloku --- */
        .explain {{
            background: #FFFBF2;
            border-left: 5px solid {GOLD};
            border-radius: 0 10px 10px 0;
            padding: 0.85rem 1.1rem;
            margin: 0.5rem 0 1.7rem 0;
            font-size: 0.95rem;
            color: #4A3F46;
            line-height: 1.55;
        }}
        .explain b {{ color: {BURGUNDY}; }}

        /* --- Ümumi kart --- */
        .card {{
            background: #FFFFFF;
            border-radius: 13px;
            padding: 1.15rem 1.3rem;
            box-shadow: 0 2px 10px rgba(36,26,34,0.08);
            border-left: 5px solid {TEAL};
            margin-bottom: 1rem;
            height: 100%;
        }}
        .card h4 {{ margin: 0 0 0.45rem 0; color: {BURGUNDY}; font-size: 1.05rem; }}
        .card p {{ margin: 0; color: #4A3F46; font-size: 0.93rem; line-height: 1.55; }}

        /* --- Mərhələ addımları --- */
        .step {{
            background: #FFFFFF; border-radius: 12px; padding: 0.95rem 1.1rem;
            box-shadow: 0 2px 9px rgba(36,26,34,0.08);
            border-top: 4px solid {TEAL}; height: 100%;
        }}
        .step .num {{
            display:inline-block; width: 26px; height: 26px; line-height: 26px;
            border-radius: 50%; background: {GOLD}; color: #3A2A10;
            text-align:center; font-weight: 800; font-size: 0.85rem;
            margin-bottom: 0.4rem;
        }}
        .step h5 {{ margin: 0.25rem 0 0.3rem 0; color: {BURGUNDY}; font-size: 0.98rem; }}
        .step p {{ margin: 0; font-size: 0.87rem; color: #564B52; line-height: 1.5; }}

        /* --- Sintetik verilən xəbərdarlığı --- */
        .synthetic {{
            background: rgba(201,150,43,0.14);
            border: 1px dashed {GOLD};
            border-radius: 9px;
            padding: 0.55rem 0.8rem;
            font-size: 0.83rem;
            color: #6B5320;
            text-align: center;
            margin-bottom: 0.9rem;
        }}

        /* --- Model etiketləri --- */
        .tag {{
            display:inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
            font-size: 0.78rem; font-weight: 700; color: #fff;
        }}

        /* --- Cədvəl --- */
        .stDataFrame {{ border-radius: 10px; overflow: hidden; }}

        /* --- Yan panel --- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FBF6EE 0%, #F2E7D8 100%);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    """Səhifənin yuxarısındakı rəngli başlıq."""
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def page_setup(title: str, subtitle: str) -> None:
    """Hər səhifənin ilk çağırdığı funksiya."""
    inject_css()
    sidebar_note()
    hero(title, subtitle)


def sidebar_note() -> None:
    """Yan paneldə həmişə görünən sintetik verilən xəbərdarlığı."""
    with st.sidebar:
        st.markdown(
            f'<div class="synthetic">⚠️ {config.SYNTHETIC_NOTE}</div>',
            unsafe_allow_html=True,
        )


def explain(text: str) -> None:
    """Diaqramın altındakı "Bu nə deməkdir?" izahı."""
    st.markdown(
        f'<div class="explain"><b>Bu nə deməkdir?</b> {text}</div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, hint: str = "") -> None:
    """Bir KPI kartı."""
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str, color: str = TEAL) -> None:
    """Başlıqlı məlumat kartı."""
    st.markdown(
        f'<div class="card" style="border-left-color:{color}">'
        f"<h4>{title}</h4><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def model_tag(key: str) -> str:
    """Model adını öz rəngində etiket kimi qaytarır (HTML)."""
    color = config.MODEL_COLORS.get(key, SLATE)
    name = config.MODEL_SHORT_AZ.get(key, key)
    return f'<span class="tag" style="background:{color}">{name}</span>'


def synthetic_footer() -> None:
    """Səhifənin sonundakı xəbərdarlıq."""
    st.markdown(
        f'<div class="synthetic">{config.SYNTHETIC_NOTE}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# PLOTLY ÜÇÜN ÜMUMİ GÖRÜNÜŞ
# --------------------------------------------------------------------------

def style_fig(fig: go.Figure, height: int = 400, showlegend: bool = True) -> go.Figure:
    """Bütün diaqramlara eyni kino görünüşünü verir."""
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        font=dict(family="Segoe UI, sans-serif", size=13, color=INK),
        title_font=dict(size=16, color=BURGUNDY),
        margin=dict(l=55, r=25, t=55, b=50),
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor="rgba(107,116,137,0.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(107,116,137,0.18)", zeroline=False)
    return fig


def show(fig: go.Figure, note: str, height: int = 400, showlegend: bool = True) -> None:
    """Diaqramı çəkir və altında izahını yazır."""
    st.plotly_chart(
        style_fig(fig, height=height, showlegend=showlegend),
        width="stretch",
        config={"displayModeBar": False},
    )
    explain(note)


# --------------------------------------------------------------------------
# KEŞLƏNMİŞ VERİLƏNLƏR
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CSV faylları (yalnız bir dəfə oxunur)."""
    return load_all()


@st.cache_data(show_spinner=False)
def get_split():
    """Zamana görə bölgü (yalnız bir dəfə hesablanır)."""
    _, _, ratings = get_data()
    return split_ratings(ratings)


@st.cache_data(show_spinner=False)
def get_titles() -> dict[int, str]:
    _, films, _ = get_data()
    return film_title_map(films)


@st.cache_data(show_spinner=False)
def get_results() -> dict:
    """results/ qovluğundakı hazır nəticələr.

    DİQQƏT: tətbiq heç bir metrikanı özü hesablamır - yalnız
    `python -m src.experiments` ilə hesablanmış faylları oxuyur.
    """
    enc = config.CSV_ENCODING
    out: dict = {}
    for name in (
        "model_comparison",
        "k_sweep",
        "cold_start",
        "epoch_sweep",
        "svd_training_curve",
        "split_summary",
    ):
        path = config.RESULTS_DIR / f"{name}.csv"
        out[name] = pd.read_csv(path, encoding=enc) if path.exists() else None

    meta_path = config.RESULTS_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as handle:
            out["meta"] = json.load(handle)
    else:
        out["meta"] = {}
    return out


def results_ready() -> bool:
    """Nəticə faylları mövcuddurmu?"""
    return (config.RESULTS_DIR / "model_comparison.csv").exists()


def require_results() -> dict | None:
    """Nəticələr yoxdursa, istifadəçiyə nə etməli olduğunu izah edir."""
    if not results_ready():
        st.error(
            "Nəticə faylları tapılmadı. Əvvəlcə terminalda bu əmri işlədin:\n\n"
            "`python -m src.experiments`"
        )
        return None
    return get_results()


# --------------------------------------------------------------------------
# KEŞLƏNMİŞ MODELLƏR
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_popularity() -> PopularityRecommender:
    split = get_split()
    return PopularityRecommender().fit(split.train_full)


@st.cache_resource(show_spinner=False)
def get_item_cf() -> ItemCFRecommender:
    split = get_split()
    _, films, _ = get_data()
    return ItemCFRecommender().fit(
        split.train_full, all_film_ids=films["film_id"].to_numpy()
    )


@st.cache_resource(show_spinner=False)
def get_svd(n_factors: int, n_epochs: int = config.SVD_EPOCHS) -> SVDRecommender:
    """SVD modeli. Hər (k, epoch) cütü üçün bir dəfə öyrədilir və keşlənir."""
    split = get_split()
    _, films, _ = get_data()
    return SVDRecommender(n_factors=n_factors, n_epochs=n_epochs).fit(
        split.train_full, all_film_ids=films["film_id"].to_numpy()
    )


@st.cache_data(show_spinner=False)
def get_candidate_pool() -> np.ndarray:
    """Təlimdə ən azı 3 reytinq almış filmlər."""
    return ev.eligible_films(get_split().train_full)


@st.cache_data(show_spinner=False)
def get_seen_films() -> dict[int, set[int]]:
    """Hər müştərinin təlimdə gördüyü filmlər."""
    return ev.seen_films_by_customer(get_split().train_full)


def candidates_for(customer_id: int) -> np.ndarray:
    """Bir müştəri üçün namizəd filmlər (əvvəl görmədikləri)."""
    seen = get_seen_films().get(int(customer_id), set())
    return np.array([f for f in get_candidate_pool() if f not in seen])


@st.cache_data(show_spinner=False)
def get_svd_rmse(n_factors: int, n_epochs: int = config.SVD_EPOCHS) -> float:
    """Seçilmiş k üçün test RMSE (slayder dəyişdikdə yenilənir)."""
    return ev.rmse(get_svd(n_factors, n_epochs), get_split().test)
