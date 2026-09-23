"""
Səhifə 6: SOYUQ START
=====================

İki hissə:
  A) Canlı təcrübə - siz özünüz 3-5 filmə ulduz verirsiniz və dərhal
     tövsiyə alırsınız (yeni istifadəçi kimi).
  B) Ölçülmüş nəticə - 30 müştərinin reytinqləri 3 / 5 / 10-a qədər
     azaldıldıqda Precision@10 necə dəyişir.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src import ui

ui.page_setup(
    "❄️ Soyuq start",
    "Sistem sizi tanımırsa nə edir? Özünüz sınayın və ölçülmüş nəticələrə baxın",
)

customers, films, ratings = ui.get_data()
split = ui.get_split()
titles = ui.get_titles()
results = ui.require_results()
if results is None:
    st.stop()

meta = results["meta"]
cold_start = results["cold_start"]
film_info = films.set_index("film_id")

st.markdown(
    """
**Soyuq start problemi nədir?** Sistemə yeni qeydiyyatdan keçmiş bir müştəri
haqqında heç nə bilinmir: nə baxdığı, nə bəyəndiyi. Əməkdaşlıq süzgəci və SVD
isə məhz keçmiş reytinqlərdən qidalanır. Bəs onda nə etməli?
"""
)

st.divider()

# ==========================================================================
# A) CANLI TƏCRÜBƏ
# ==========================================================================

st.markdown("### 🧪 A. Özünüz sınayın")
st.caption(
    "Aşağıdakı filmlərdən ən azı 3-nə ulduz verin. Sistem sizi yeni müştəri "
    "kimi qəbul edəcək və dərhal tövsiyə hazırlayacaq."
)

# Ən populyar filmləri təklif edirik - yeni istifadəçinin tanıma ehtimalı yüksəkdir
popularity_model = ui.get_popularity()
top_films = [film_id for film_id, _ in popularity_model.top_films(n=40)]

rng = np.random.default_rng(config.SEED)
if "cold_start_choices" not in st.session_state:
    st.session_state["cold_start_choices"] = [
        int(f) for f in rng.choice(top_films, size=8, replace=False)
    ]

choice_ids = st.session_state["cold_start_choices"]

STAR_OPTIONS = {
    0: "— qiymətləndirmədim",
    1: "⭐",
    2: "⭐⭐",
    3: "⭐⭐⭐",
    4: "⭐⭐⭐⭐",
    5: "⭐⭐⭐⭐⭐",
}

given: list[tuple[int, float]] = []
columns = st.columns(4)
for index, film_id in enumerate(choice_ids):
    row = film_info.loc[film_id]
    with columns[index % 4]:
        value = st.selectbox(
            f"**{row['title']}**",
            options=list(STAR_OPTIONS),
            format_func=lambda v: STAR_OPTIONS[v],
            key=f"star_{film_id}",
            help=f"{row['year']} • {row['genres']}",
        )
        if value > 0:
            given.append((int(film_id), float(value)))

button_column, reset_column = st.columns([1, 4])
with button_column:
    if st.button("🔄 Başqa filmlər göstər"):
        new_rng = np.random.default_rng()
        st.session_state["cold_start_choices"] = [
            int(f) for f in new_rng.choice(top_films, size=8, replace=False)
        ]
        for film_id in choice_ids:
            st.session_state.pop(f"star_{film_id}", None)
        st.rerun()

st.write("")

if len(given) < 3:
    st.info(
        f"Hələ **{len(given)}** filmə qiymət vermisiniz. Tövsiyə almaq üçün "
        f"ən azı **3** lazımdır.",
        icon="⭐",
    )
else:
    rated_ids = {film_id for film_id, _ in given}
    candidates = np.array([f for f in ui.get_candidate_pool() if f not in rated_ids])

    item_cf_model = ui.get_item_cf()
    svd_model = ui.get_svd(int(meta.get("best_k", config.SVD_DEFAULT_FACTORS)))

    st.success(
        f"**{len(given)} reytinq qəbul edildi.** Aşağıda hər üç modelin "
        f"sizin üçün hazırladığı siyahı var.",
        icon="✅",
    )

    recommendation_sets = {
        "popularity": popularity_model.recommend(-1, candidates, n=5),
        "item_cf": item_cf_model.recommend_new_user(given, candidates, n=5),
        "svd": svd_model.recommend_foldin(given, candidates, n=5),
    }

    columns = st.columns(3)
    for column, key in zip(columns, ("popularity", "item_cf", "svd")):
        color = config.MODEL_COLORS[key]
        with column:
            st.markdown(
                f'<div style="background:{color};color:#fff;border-radius:11px;'
                f'padding:0.7rem 0.9rem;margin-bottom:0.7rem;">'
                f'<b>{config.MODEL_NAMES_AZ[key]}</b></div>',
                unsafe_allow_html=True,
            )
            for rank, (film_id, score) in enumerate(recommendation_sets[key], start=1):
                row = film_info.loc[film_id]
                score_text = (
                    f"{int(score)} reytinq" if key == "popularity"
                    else f"proqnoz {score:.2f}"
                )
                st.markdown(
                    f'<div style="background:#fff;border:1px solid '
                    f'rgba(107,116,137,0.22);border-radius:9px;'
                    f'padding:0.5rem 0.7rem;margin-bottom:0.35rem;">'
                    f'<div style="font-size:0.9rem;font-weight:700;">'
                    f'{rank}. {row["title"]}</div>'
                    f'<div style="font-size:0.76rem;color:#6B6169">'
                    f'{row["year"]} • {row["genres"]} • {score_text}</div></div>',
                    unsafe_allow_html=True,
                )

    ui.explain(
        "<b>Populyarlıq</b> sizin ballarınıza ümumiyyətlə baxmır - siyahı "
        "kim olursa olsun eynidir. <b>Item-CF</b> verdiyiniz ballara oxşar "
        "filmləri hazır oxşarlıq matrisində tapır. <b>SVD</b> isə “fold-in” "
        "üsulu ilə yalnız <b>sizin</b> gizli vektorunuzu öyrədir: film "
        "vektorları dondurulur, ona görə bütün model yenidən öyrədilmir və "
        "cavab dərhal gəlir."
    )

    with st.expander("🔍 SVD sizin haqqınızda nə “öyrəndi”?"):
        bias, vector = svd_model.fold_in(given)
        st.markdown(
            f"**Sizin meyliniz (b_u): `{bias:+.3f}`** — "
            + (
                "müsbətdir, yəni ortalama müştəridən <b>səxavətli</b> "
                "qiymətləndirirsiniz."
                if bias > 0
                else "mənfidir, yəni ortalama müştəridən <b>sərt</b> "
                "qiymətləndirirsiniz."
            ),
            unsafe_allow_html=True,
        )
        vector_fig = go.Figure(
            go.Bar(
                x=[f"f{i + 1}" for i in range(len(vector))],
                y=vector,
                marker_color=[
                    ui.BURGUNDY if v >= 0 else ui.SLATE for v in vector
                ],
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            )
        )
        vector_fig.update_layout(
            title="Sizin gizli zövq vektorunuz (p_u)",
            xaxis_title="Gizli faktor",
            yaxis_title="Dəyər",
        )
        ui.show(
            vector_fig,
            "Bu sütunların <b>adı yoxdur</b> - model onları verilənlərdən özü "
            "kəşf edib. Müsbət sütun “bu gizli xüsusiyyəti sevirəm”, mənfi "
            "sütun isə “bundan qaçıram” deməkdir. Filmin vektoru ilə sizinki "
            "üst-üstə düşəndə proqnoz balı yüksəlir.",
            height=300,
            showlegend=False,
        )

st.divider()

# ==========================================================================
# B) ÖLÇÜLMÜŞ NƏTİCƏ
# ==========================================================================

st.markdown("### 📉 B. Ölçülmüş nəticə: az məlumat nəyə başa gəlir?")

st.markdown(
    f"""
**Təcrübənin qurulması:** {config.COLD_START_N_CUSTOMERS} təsadüfi müştəri
seçildi və onların təlim reytinqləri süni şəkildə yalnız **ilk 3, 5 və 10**
ədədə endirildi (digər müştərilərin verilənləri toxunulmadı). Sonra hər üç
model yenidən öyrədildi və məhz bu müştərilər üçün Precision@10 ölçüldü.
"""
)

ordered = cold_start.sort_values("sira")
labels = {"3": "3 reytinq", "5": "5 reytinq", "10": "10 reytinq", "Tam": "Tam təlim"}

fig = go.Figure()
for key in ("popularity", "item_cf", "svd"):
    subset = ordered[ordered["model_key"] == key]
    fig.add_trace(
        go.Scatter(
            x=[labels.get(str(v), str(v)) for v in subset["reytinq_sayi"]],
            y=subset["precision_at_10"],
            mode="lines+markers",
            name=config.MODEL_NAMES_AZ[key],
            line=dict(color=config.MODEL_COLORS[key], width=3.5),
            marker=dict(size=11),
            hovertemplate="%{x}<br>P@10: %{y:.4f}<extra></extra>",
        )
    )
fig.update_layout(
    title="Təlim reytinqlərinin sayı → Precision@10",
    xaxis_title="Müştəri haqqında bilinən reytinq sayı",
    yaxis_title="Precision@10",
)

evaluated = int(cold_start["musteri_sayi"].iloc[0])
ui.show(
    fig,
    f"Hər üç əyri soldan sağa <b>yuxarı qalxır</b>: müştəri haqqında nə qədər "
    f"çox bilsək, tövsiyə bir o qədər yaxşılaşır. Ən vacib müşahidə odur ki, "
    f"<b>ən az məlumat olan nöqtədə (3 reytinq) populyarlıq modeli öndədir</b> - "
    f"çünki o, müştərini tanımağa ehtiyac duymur. Qeyd: ölçmə "
    f"{evaluated} müştəri üzrə aparılıb, ona görə əyrilərdə kiçik dalğalanmalar "
    f"təsadüfi ola bilər.",
    height=420,
)

# --- Rəqəm cədvəli ---
pivot = (
    cold_start.pivot_table(
        index="sira", columns="model_key", values="precision_at_10"
    )
    .rename(index={3: "3 reytinq", 5: "5 reytinq", 10: "10 reytinq",
                   999: "Tam təlim"})
    .rename(columns=config.MODEL_SHORT_AZ)
)
pivot.index.name = "Bilinən reytinq"
st.dataframe(
    pivot.style.format("{:.4f}").highlight_max(axis=1, color="#D7EBE3"),
    use_container_width=True,
)

full_row = cold_start[cold_start["sira"] == 999].set_index("model_key")
three_row = cold_start[cold_start["sira"] == 3].set_index("model_key")

drops = []
for key in ("popularity", "item_cf", "svd"):
    full_value = full_row.loc[key, "precision_at_10"]
    small_value = three_row.loc[key, "precision_at_10"]
    if full_value and not np.isnan(full_value) and full_value > 0:
        drops.append((key, 1 - small_value / full_value))

st.markdown("#### 🔑 Nə öyrəndik?")
for key, drop in sorted(drops, key=lambda item: item[1]):
    st.markdown(
        f"- **{config.MODEL_NAMES_AZ[key]}**: 3 reytinqlə tam təlimə nisbətən "
        f"Precision@10 **{drop:.0%}** aşağıdır."
    )

st.success(
    "**Praktiki tövsiyə:** real sistemdə yeni istifadəçiyə əvvəlcə "
    "**populyar filmlər** göstərilməlidir. İstifadəçi 10-15 film "
    "qiymətləndirdikdən sonra fərdiləşdirilmiş modellərə keçmək olar. "
    "Bu yanaşmaya **hibrid strategiya** deyilir.",
    icon="💡",
)

ui.synthetic_footer()
