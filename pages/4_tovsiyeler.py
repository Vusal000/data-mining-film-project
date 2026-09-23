"""
Səhifə 5: MÜŞTƏRİ TÖVSİYƏLƏRİ
=============================

Seçilmiş bir müştəri üçün üç modelin Top-10 siyahısı yan-yana.
Slayder ilə gizli faktor sayını (k) dəyişdikdə SVD yenidən öyrədilir -
müdafiə zamanı parametrin təsirini canlı göstərmək üçün.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src import ui

ui.page_setup(
    "🎯 Müştəri tövsiyələri",
    "Bir müştəri seçin: profilini, üç modelin təklifini və hər təklifin "
    "izahını görün",
)

customers, films, ratings = ui.get_data()
split = ui.get_split()
titles = ui.get_titles()
results = ui.get_results()
meta = results.get("meta", {})

film_info = films.set_index("film_id")

# --------------------------------------------------------------------------
# MÜŞTƏRİ SEÇİMİ VƏ k SLAYDERİ
# --------------------------------------------------------------------------

selector, slider_column = st.columns([1.5, 1])

with selector:
    options = customers.sort_values("customer_id")
    labels = {
        int(row.customer_id): f"#{row.customer_id} — {row.name} ({row.city})"
        for row in options.itertuples()
    }
    customer_id = st.selectbox(
        "Müştəri seçin",
        options=list(labels),
        format_func=lambda cid: labels[cid],
        index=0,
    )

with slider_column:
    n_factors = st.slider(
        "SVD gizli faktor sayı (k)",
        min_value=2,
        max_value=50,
        value=int(meta.get("best_k", config.SVD_DEFAULT_FACTORS)),
        step=1,
        help="Dəyişdirdikdə SVD modeli yenidən öyrədilir və aşağıdakı "
             "tövsiyələr ilə RMSE yenilənir.",
    )

customer = customers.loc[customers["customer_id"] == customer_id].iloc[0]

# Bu müştərinin təlim və test reytinqləri
train_ratings = split.train_full[split.train_full["customer_id"] == customer_id]
test_ratings = split.test[split.test["customer_id"] == customer_id]
liked_in_test = set(
    test_ratings.loc[test_ratings["rating"] >= config.LIKE_THRESHOLD, "film_id"]
    .astype(int)
)

# --------------------------------------------------------------------------
# PROFİL
# --------------------------------------------------------------------------

st.markdown("### 👤 Müştəri profili")

c1, c2, c3, c4 = st.columns(4)
with c1:
    ui.kpi("Ad", customer["name"], f"#{customer_id}")
with c2:
    ui.kpi("Şəhər", customer["city"], f"Yaş qrupu: {customer['age_group']}")
with c3:
    ui.kpi("Təlim reytinqi", str(len(train_ratings)),
           f"orta bal {train_ratings['rating'].mean():.2f}")
with c4:
    ui.kpi("Test reytinqi", str(len(test_ratings)),
           f"{len(liked_in_test)} ədədi bəyənilib (≥4)")

left, right = st.columns([1, 1.1])

# --- Radar: janr zövqü ---
with left:
    train_with_genres = (
        train_ratings.merge(films, on="film_id").explode("genre_list")
    )
    watched = train_with_genres["genre_list"].value_counts()
    loved = (
        train_with_genres[train_with_genres["rating"] >= config.LIKE_THRESHOLD][
            "genre_list"
        ].value_counts()
    )

    genres = config.GENRES
    watched_share = [100 * watched.get(g, 0) / max(watched.sum(), 1) for g in genres]
    loved_share = [100 * loved.get(g, 0) / max(loved.sum(), 1) for g in genres]

    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=watched_share + watched_share[:1],
            theta=genres + genres[:1],
            name="Baxdığı filmlər",
            line=dict(color=ui.SLATE, width=2),
            fillcolor="rgba(107,116,137,0.18)",
            fill="toself",
            hovertemplate="%{theta}<br>Baxdıqlarının %{r:.0f}%-i<extra></extra>",
        )
    )
    radar.add_trace(
        go.Scatterpolar(
            r=loved_share + loved_share[:1],
            theta=genres + genres[:1],
            name="Bəyəndiyi filmlər (≥4)",
            line=dict(color=ui.BURGUNDY, width=3),
            fillcolor="rgba(122,31,61,0.25)",
            fill="toself",
            hovertemplate="%{theta}<br>Bəyəndiklərinin %{r:.0f}%-i<extra></extra>",
        )
    )
    radar.update_layout(
        title="Janr zövqü",
        polar=dict(
            bgcolor="rgba(255,255,255,0.7)",
            radialaxis=dict(visible=True, ticksuffix="%",
                            gridcolor="rgba(107,116,137,0.25)"),
        ),
    )
    top_genres = [g for g, _ in loved.head(2).items()] or ["—"]
    ui.show(
        radar,
        f"Boz sahə müştərinin ümumiyyətlə <b>baxdığı</b>, bordo sahə isə "
        f"<b>bəyəndiyi</b> filmlərin janr payıdır. Bordo sahənin boz sahədən "
        f"kənara çıxdığı yerlər həqiqi zövqü göstərir. Bu müştəri üçün ən "
        f"güclü janr(lar): <b>{', '.join(top_genres)}</b>.",
        height=400,
    )

# --- Bəyəndiyi filmlər ---
with right:
    st.markdown("#### ❤️ Təlim dövründə bəyəndiyi filmlər")
    liked_train = (
        train_ratings[train_ratings["rating"] >= config.LIKE_THRESHOLD]
        .sort_values(["rating", "timestamp"], ascending=[False, False])
        .head(10)
        .merge(films, on="film_id")
    )
    if liked_train.empty:
        st.info("Bu müştəri təlim dövründə 4 və ya 5 bal verməyib.")
    else:
        table = pd.DataFrame(
            {
                "Film": liked_train["title"],
                "İl": liked_train["year"],
                "Janr": liked_train["genres"],
                "Bal": liked_train["rating"].map(lambda r: "⭐" * int(r)),
            }
        )
        st.dataframe(table, width="stretch", hide_index=True, height=395)
    ui.explain(
        "Modellər məhz bu tarixçəyə baxaraq tövsiyə qurur. Item-CF bu "
        "filmlərə <b>oxşar</b> filmləri axtarır, SVD isə bunlardan müştərinin "
        "<b>gizli zövq vektorunu</b> çıxarır."
    )

st.divider()

# --------------------------------------------------------------------------
# TÖVSİYƏLƏR
# --------------------------------------------------------------------------

st.markdown("### 🍿 Üç modelin Top-10 siyahısı")

with st.spinner(f"SVD modeli k={n_factors} ilə öyrədilir..."):
    models = {
        "popularity": ui.get_popularity(),
        "item_cf": ui.get_item_cf(),
        "svd": ui.get_svd(int(n_factors)),
    }
    svd_rmse = ui.get_svd_rmse(int(n_factors))

candidates = ui.candidates_for(customer_id)

info_left, info_right = st.columns([1, 1])
with info_left:
    st.markdown(
        f"**Namizəd film sayı:** {len(candidates)} "
        f"(təlimdə ən azı {config.MIN_TRAIN_RATINGS} reytinqi olan və bu "
        f"müştərinin görmədiyi filmlər)"
    )
with info_right:
    st.markdown(
        f"**SVD test RMSE (k={n_factors}):** `{svd_rmse:.4f}` &nbsp;•&nbsp; "
        f"seçilmiş k = {meta.get('best_k')}"
    )

if len(liked_in_test) == 0:
    st.warning(
        "Bu müştərinin test dövründə 4+ bal verdiyi film yoxdur, ona görə "
        "✓ işarəsi görünməyəcək. Başqa müştəri seçə bilərsiniz.",
        icon="ℹ️",
    )

columns = st.columns(3)
for column, key in zip(columns, ("popularity", "item_cf", "svd")):
    model = models[key]
    color = config.MODEL_COLORS[key]
    with column:
        subtitle = f"k = {n_factors}" if key == "svd" else "&nbsp;"
        st.markdown(
            f'<div style="background:{color};color:#fff;border-radius:11px;'
            f'padding:0.7rem 0.9rem;margin-bottom:0.7rem;">'
            f'<b>{config.MODEL_NAMES_AZ[key]}</b><br>'
            f'<span style="font-size:0.8rem;opacity:0.9">{subtitle}</span></div>',
            unsafe_allow_html=True,
        )

        recommendations = model.recommend(customer_id, candidates, n=config.TOP_N)
        hits = 0
        for rank, (film_id, score) in enumerate(recommendations, start=1):
            row = film_info.loc[film_id]
            is_hit = film_id in liked_in_test
            hits += int(is_hit)

            mark = " ✓" if is_hit else ""
            background = "#EAF6F0" if is_hit else "#FFFFFF"
            border = "#2E7D5B" if is_hit else "rgba(107,116,137,0.22)"
            score_text = (
                f"{int(score)} reytinq" if key == "popularity"
                else f"proqnoz {score:.2f}"
            )

            st.markdown(
                f'<div style="background:{background};border:1px solid {border};'
                f'border-radius:9px;padding:0.5rem 0.7rem;margin-bottom:0.35rem;">'
                f'<div style="font-size:0.9rem;font-weight:700;color:{ui.INK}">'
                f'{rank}. {row["title"]}{mark}</div>'
                f'<div style="font-size:0.76rem;color:#6B6169">'
                f'{row["year"]} • {row["genres"]} • {score_text}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="text-align:center;padding:0.45rem;border-radius:8px;'
            f'background:rgba(201,150,43,0.16);font-size:0.85rem;font-weight:700;'
            f'color:#6B5320;margin-top:0.3rem;">'
            f"Tuş gələn: {hits} / {config.TOP_N}</div>",
            unsafe_allow_html=True,
        )

ui.explain(
    "✓ işarəsi o deməkdir ki, müştəri həmin filmi <b>test dövründə həqiqətən "
    "4 və ya 5 balla qiymətləndirib</b> - yəni tövsiyə doğru çıxıb. "
    "Qeyd: ✓ olmayan film mütləq pis tövsiyə deyil; müştəri sadəcə həmin "
    "filmə hələ baxmamış ola bilər. Offline qiymətləndirmənin əsas "
    "məhdudiyyəti məhz budur."
)

st.divider()

# --------------------------------------------------------------------------
# İZAHLAR
# --------------------------------------------------------------------------

st.markdown("### 💬 Tövsiyələrin izahı")
st.caption(
    "Hər model öz “düşüncə tərzini” izah edir. Real sistemlərdə izah "
    "istifadəçi etibarını əhəmiyyətli dərəcədə artırır."
)

tabs = st.tabs([config.MODEL_NAMES_AZ[key] for key in ("popularity", "item_cf", "svd")])
for tab, key in zip(tabs, ("popularity", "item_cf", "svd")):
    with tab:
        model = models[key]
        for rank, (film_id, _) in enumerate(
            model.recommend(customer_id, candidates, n=5), start=1
        ):
            row = film_info.loc[film_id]
            st.markdown(
                f'<div style="border-left:4px solid {config.MODEL_COLORS[key]};'
                f'background:#fff;border-radius:0 9px 9px 0;padding:0.6rem 0.9rem;'
                f'margin-bottom:0.45rem;">'
                f'<b>{rank}. {row["title"]}</b> '
                f'<span style="color:#6B6169;font-size:0.82rem">({row["year"]})</span>'
                f'<br><span style="font-size:0.88rem;color:#4A3F46">'
                f"{model.explain(customer_id, film_id, titles)}</span></div>",
                unsafe_allow_html=True,
            )

st.divider()

# --------------------------------------------------------------------------
# k-nın CANLI TƏSİRİ
# --------------------------------------------------------------------------

st.markdown("### 🎚️ Slayderin təsiri: k dəyişdikdə nə olur?")

overlap_rows = []
reference = {
    film_id for film_id, _ in models["svd"].recommend(customer_id, candidates, n=10)
}
for k_value in config.SVD_FACTORS_GRID:
    other = {
        film_id
        for film_id, _ in ui.get_svd(int(k_value)).recommend(
            customer_id, candidates, n=10
        )
    }
    overlap_rows.append(
        {
            "k": k_value,
            "ortaq_film": len(reference & other),
            "rmse": ui.get_svd_rmse(int(k_value)),
        }
    )
overlap = pd.DataFrame(overlap_rows)

left, right = st.columns(2)
with left:
    fig = go.Figure(
        go.Bar(
            x=[str(k) for k in overlap["k"]],
            y=overlap["ortaq_film"],
            marker_color=ui.BURGUNDY,
            text=overlap["ortaq_film"],
            textposition="outside",
            hovertemplate="k=%{x}<br>Ortaq film: %{y}/10<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"k={n_factors} siyahısı ilə ortaq film sayı",
        xaxis_title="Müqayisə edilən k",
        yaxis_title="Ortaq film (10-dan)",
        yaxis_range=[0, 10.6],
    )
    ui.show(
        fig,
        f"Hazırkı seçiminizlə (k={n_factors}) digər k dəyərlərinin Top-10 "
        "siyahısı arasında neçə ortaq film olduğunu göstərir. Ortaq film nə "
        "qədər azdırsa, k parametri tövsiyələri bir o qədər çox dəyişir - "
        "<b>slayderi tərpədib bu diaqramın necə dəyişdiyini müdafiədə canlı "
        "göstərə bilərsiniz</b>.",
        height=350,
        showlegend=False,
    )

with right:
    fig = go.Figure(
        go.Scatter(
            x=overlap["k"],
            y=overlap["rmse"],
            mode="lines+markers",
            line=dict(color=ui.TEAL, width=3),
            marker=dict(size=10),
            hovertemplate="k=%{x}<br>Test RMSE: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=n_factors, line_dash="dash", line_color=ui.GOLD,
        annotation_text=f"seçdiyiniz k={n_factors}", annotation_position="top",
    )
    fig.update_layout(
        title="k → test RMSE",
        xaxis_title="Gizli faktor sayı (k)",
        yaxis_title="Test RMSE",
        xaxis_type="log",
    )
    fig.update_xaxes(tickvals=config.SVD_FACTORS_GRID,
                     ticktext=[str(k) for k in config.SVD_FACTORS_GRID])
    ui.show(
        fig,
        f"Tapşırıqda tələb olunan {config.SVD_EPOCHS} epoch ayarı ilə RMSE "
        "k-dan az asılıdır - səbəbi “Model müqayisəsi” səhifəsindəki epoch "
        "diaqnozudur: model bu qədər epoch-da hələ tam öyrənmir, ona görə "
        "əlavə gizli faktorlar işə düşmür.",
        height=350,
        showlegend=False,
    )

ui.synthetic_footer()
