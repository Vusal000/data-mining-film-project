"""
Səhifə 2: VERİLƏNLƏR
====================

Verilənləri modelə vermədən ƏVVƏL onları tanımaq lazımdır.
Bu səhifə yeddi suala cavab verir: matris nə qədər boşdur, ballar necə
paylanıb, populyarlıq necə davranır, janrlar, müştəri aktivliyi,
şəhər/yaş fərqləri və zaman dinamikası.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src import ui

ui.page_setup(
    "📊 Verilənlərin təhlili",
    "Modeldən əvvəl verilənləri tanıyırıq: seyrəklik, paylanmalar, "
    "uzun quyruq və müştəri davranışı",
)

customers, films, ratings = ui.get_data()

# Reytinqləri müştəri və film məlumatları ilə birləşdiririk
merged = ratings.merge(customers, on="customer_id").merge(films, on="film_id")

# --------------------------------------------------------------------------
# 1. REYTİNQ MATRİSİNİN İSTİLİK XƏRİTƏSİ
# --------------------------------------------------------------------------

st.markdown("### 1. Reytinq matrisi: sistemin əsas problemi")

# Müştəriləri aktivliyə, filmləri populyarlığa görə sıralayırıq ki,
# strukturu görmək asan olsun
customer_order = ratings.groupby("customer_id").size().sort_values(ascending=False).index
film_order = ratings.groupby("film_id").size().sort_values(ascending=False).index

matrix = (
    ratings.pivot(index="customer_id", columns="film_id", values="rating")
    .reindex(index=customer_order, columns=film_order)
)

heatmap = go.Figure(
    go.Heatmap(
        z=matrix.to_numpy(),
        colorscale=[
            [0.0, "#F4ECE1"],
            [0.25, "#D9C29A"],
            [0.5, ui.GOLD],
            [0.75, "#A8452F"],
            [1.0, ui.BURGUNDY],
        ],
        zmin=1,
        zmax=5,
        hovertemplate="Müştəri sırası: %{y}<br>Film sırası: %{x}<br>Bal: %{z}<extra></extra>",
        colorbar=dict(title="Bal", thickness=14),
    )
)
heatmap.update_layout(
    title="Müştəri × Film reytinq matrisi (boş yerlər = reytinq yoxdur)",
    xaxis_title="Filmlər (populyarlığa görə sıralanıb →)",
    yaxis_title="Müştərilər (aktivliyə görə ↓)",
)
heatmap.update_xaxes(showticklabels=False)
heatmap.update_yaxes(showticklabels=False)

sparsity = 1 - len(ratings) / (len(customers) * len(films))
ui.show(
    heatmap,
    f"Rəngli nöqtə = mövcud reytinq, açıq fon = <b>boş xana</b>. Matrisin "
    f"<b>{sparsity:.1%}</b>-i boşdur. Sol yuxarı küncdə rəng daha sıxdır: "
    "aktiv müştərilər populyar filmlərə baxır. Sağ aşağı künc demək olar "
    "tamamilə boşdur - <b>modelin doldurmalı olduğu yer məhz buradır</b>.",
    height=460,
    showlegend=False,
)

st.divider()

# --------------------------------------------------------------------------
# 2. BAL PAYLANMASI + 3. UZUN QUYRUQ
# --------------------------------------------------------------------------

left, right = st.columns(2)

with left:
    st.markdown("### 2. Ballar necə paylanıb?")
    distribution = ratings["rating"].value_counts().sort_index()
    bars = go.Figure(
        go.Bar(
            x=[f"{star} ulduz" for star in distribution.index],
            y=distribution.to_numpy(),
            marker_color=[ui.SLATE, "#8B7F6B", ui.GOLD, "#A8452F", ui.BURGUNDY],
            text=[f"{v / len(ratings):.1%}" for v in distribution],
            textposition="outside",
            hovertemplate="%{x}<br>%{y} reytinq<extra></extra>",
        )
    )
    bars.update_layout(
        title=f"Bal paylanması (orta: {ratings['rating'].mean():.2f})",
        yaxis_title="Reytinq sayı",
    )
    liked_share = (ratings["rating"] >= config.LIKE_THRESHOLD).mean()
    ui.show(
        bars,
        f"Ballar 4 ulduz ətrafında toplanıb - insanlar adətən baxmaq "
        f"istədikləri filmi seçir, ona görə mənfi qiymət azdır. "
        f"Reytinqlərin <b>{liked_share:.1%}</b>-i “bəyənilmiş” (≥4) sayılır; "
        "Precision və Recall hesablanarkən məhz bu hədd istifadə olunur.",
        height=360,
        showlegend=False,
    )

with right:
    st.markdown("### 3. Populyarlığın “uzun quyruğu”")
    film_counts = ratings.groupby("film_id").size().sort_values(ascending=False)
    tail = go.Figure(
        go.Scatter(
            x=np.arange(1, len(film_counts) + 1),
            y=film_counts.to_numpy(),
            mode="lines",
            line=dict(color=ui.BURGUNDY, width=3),
            fill="tozeroy",
            fillcolor="rgba(122,31,61,0.16)",
            hovertemplate="Sıra: %{x}<br>Reytinq sayı: %{y}<extra></extra>",
        )
    )
    top20_share = film_counts.head(40).sum() / film_counts.sum()
    tail.update_layout(
        title="Filmlər populyarlığa görə sıralanıb",
        xaxis_title="Film sırası (1 = ən populyar)",
        yaxis_title="Reytinq sayı",
    )
    ui.show(
        tail,
        f"Klassik <b>uzun quyruq</b>: bir neçə film çox reytinq alır, "
        f"əksəriyyəti isə çox az. Ən populyar 40 film (kataloqun 20%-i) "
        f"bütün reytinqlərin <b>{top20_share:.0%}</b>-ni toplayır. Buna görə "
        "populyarlıq modeli güclü bazadır - amma o, quyruqdakı filmləri "
        "heç vaxt təklif etmir.",
        height=360,
        showlegend=False,
    )

st.divider()

# --------------------------------------------------------------------------
# 4. JANRLAR + 5. MÜŞTƏRİ AKTİVLİYİ
# --------------------------------------------------------------------------

left, right = st.columns(2)

with left:
    st.markdown("### 4. Janrların payı")
    genre_rows = films.explode("genre_list")
    genre_counts = genre_rows["genre_list"].value_counts()
    genre_ratings = (
        merged.assign(genre_list=merged["genre_list"])
        .explode("genre_list")
        .groupby("genre_list")["rating"]
        .mean()
    )

    genre_fig = go.Figure(
        go.Bar(
            y=genre_counts.index[::-1],
            x=genre_counts.to_numpy()[::-1],
            orientation="h",
            marker=dict(
                color=[genre_ratings.get(g, 0) for g in genre_counts.index[::-1]],
                colorscale=[[0, ui.SLATE], [0.5, ui.GOLD], [1, ui.BURGUNDY]],
                colorbar=dict(title="Orta<br>bal", thickness=13),
            ),
            hovertemplate="%{y}<br>%{x} film<extra></extra>",
        )
    )
    genre_fig.update_layout(
        title="Kataloqda janrların sayı (rəng = orta bal)",
        xaxis_title="Film sayı",
    )
    ui.show(
        genre_fig,
        "Sütunun uzunluğu kataloqda həmin janrda neçə film olduğunu, rəngi "
        "isə həmin janrın orta balını göstərir. Janrlar bərabər paylanmayıb - "
        "Dram və Komediya daha çoxdur, bu da real kataloqlara bənzəyir.",
        height=400,
        showlegend=False,
    )

with right:
    st.markdown("### 5. Müştəri aktivliyi")
    per_customer = ratings.groupby("customer_id").size()
    activity = go.Figure(
        go.Histogram(
            x=per_customer.to_numpy(),
            nbinsx=22,
            marker_color=ui.TEAL,
            marker_line=dict(color="white", width=1),
            hovertemplate="%{x} reytinq<br>%{y} müştəri<extra></extra>",
        )
    )
    activity.add_vline(
        x=per_customer.mean(),
        line_dash="dash",
        line_color=ui.BURGUNDY,
        annotation_text=f"orta {per_customer.mean():.0f}",
        annotation_position="top right",
    )
    activity.update_layout(
        title="Müştəri başına reytinq sayı",
        xaxis_title="Reytinq sayı",
        yaxis_title="Müştəri sayı",
    )
    ui.show(
        activity,
        f"Hər müştərinin <b>{per_customer.min()}</b> ilə "
        f"<b>{per_customer.max()}</b> arasında reytinqi var (orta "
        f"{per_customer.mean():.0f}). Az reytinqi olan müştərilər üçün "
        "proqnoz vermək daha çətindir - bu, “soyuq start” səhifəsinin mövzusudur.",
        height=400,
        showlegend=False,
    )

st.divider()

# --------------------------------------------------------------------------
# 6. ŞƏHƏR VƏ YAŞ QRUPU
# --------------------------------------------------------------------------

st.markdown("### 6. Şəhər və yaş qrupu üzrə orta bal")

left, right = st.columns(2)

with left:
    by_city = (
        merged.groupby("city")["rating"].agg(["mean", "count"]).sort_values("mean")
    )
    city_fig = go.Figure(
        go.Bar(
            x=by_city["mean"].to_numpy(),
            y=by_city.index,
            orientation="h",
            marker_color=ui.TEAL,
            text=[f"{v:.2f}" for v in by_city["mean"]],
            textposition="outside",
            customdata=by_city["count"].to_numpy(),
            hovertemplate="%{y}<br>Orta bal: %{x:.2f}<br>Reytinq: %{customdata}<extra></extra>",
        )
    )
    city_fig.update_layout(
        title="Şəhər üzrə orta bal", xaxis_title="Orta bal", xaxis_range=[3.0, 4.0]
    )
    city_spread = by_city["mean"].max() - by_city["mean"].min()
    ui.show(
        city_fig,
        f"Şəhərlər arasındakı fərq cəmi <b>{city_spread:.2f} bal</b>dır - yəni "
        "praktiki olaraq yoxdur. Bu gözləniləndir: verilənlər yaradılarkən "
        "şəhər reytinqə təsir etmir. Diaqram bunu <b>təsdiqləyir</b> və "
        "modelin şəhəri istifadə etməməsinin haqlı olduğunu göstərir.",
        height=330,
        showlegend=False,
    )

with right:
    age_order = ["18-24", "25-34", "35-44", "45-54", "55+"]
    by_age = merged.groupby("age_group")["rating"].agg(["mean", "count"])
    by_age = by_age.reindex([a for a in age_order if a in by_age.index])
    age_fig = go.Figure(
        go.Bar(
            x=by_age.index,
            y=by_age["mean"].to_numpy(),
            marker_color=ui.BURGUNDY,
            text=[f"{v:.2f}" for v in by_age["mean"]],
            textposition="outside",
            customdata=by_age["count"].to_numpy(),
            hovertemplate="%{x}<br>Orta bal: %{y:.2f}<br>Reytinq: %{customdata}<extra></extra>",
        )
    )
    age_fig.update_layout(
        title="Yaş qrupu üzrə orta bal", yaxis_title="Orta bal", yaxis_range=[3.0, 4.0]
    )
    ui.show(
        age_fig,
        "Yaş qrupları arasında da ciddi fərq yoxdur. Müştərilər arasındakı "
        "əsas fərq <b>demoqrafiyada deyil, şəxsi zövqdədir</b> - məhz buna "
        "görə tövsiyə modelləri yaş və şəhərə yox, <b>reytinq davranışına</b> "
        "baxır.",
        height=330,
        showlegend=False,
    )

st.divider()

# --------------------------------------------------------------------------
# 7. ZAMAN DİNAMİKASI
# --------------------------------------------------------------------------

st.markdown("### 7. Reytinqlərin zaman üzrə axını")

monthly = (
    ratings.set_index("timestamp")
    .resample("MS")
    .agg(reytinq_sayi=("rating", "size"), orta_bal=("rating", "mean"))
    .reset_index()
)

time_fig = go.Figure()
time_fig.add_trace(
    go.Bar(
        x=monthly["timestamp"],
        y=monthly["reytinq_sayi"],
        name="Reytinq sayı",
        marker_color=ui.TEAL,
        opacity=0.75,
        hovertemplate="%{x|%Y-%m}<br>%{y} reytinq<extra></extra>",
    )
)
time_fig.add_trace(
    go.Scatter(
        x=monthly["timestamp"],
        y=monthly["orta_bal"],
        name="Orta bal",
        yaxis="y2",
        mode="lines",
        line=dict(color=ui.BURGUNDY, width=3),
        hovertemplate="%{x|%Y-%m}<br>Orta bal: %{y:.2f}<extra></extra>",
    )
)
time_fig.update_layout(
    title="Aylar üzrə reytinq sayı və orta bal",
    xaxis_title="Tarix",
    yaxis=dict(title="Reytinq sayı"),
    yaxis2=dict(title="Orta bal", overlaying="y", side="right", range=[2.8, 4.4]),
)
ui.show(
    time_fig,
    "Reytinq sayı zamanla artır, çünki müştərilər müxtəlif tarixlərdə "
    "qeydiyyatdan keçir və kataloqa yeni filmlər əlavə olunur. Orta bal isə "
    "sabit qalır - yəni <b>zamanla “bal inflyasiyası” yoxdur</b>. Bu vacibdir: "
    "zamana görə bölgü apararkən test dövrünün sistematik fərqli olmadığını bilirik.",
    height=400,
)

# --------------------------------------------------------------------------
# XAM VERİLƏNLƏR
# --------------------------------------------------------------------------

with st.expander("🔎 Xam cədvəllərə baxış"):
    tab1, tab2, tab3 = st.tabs(["Müştərilər", "Filmlər", "Reytinqlər"])
    with tab1:
        st.dataframe(customers.head(25), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(
            films.drop(columns=["genre_list"]).head(25),
            use_container_width=True,
            hide_index=True,
        )
    with tab3:
        st.dataframe(ratings.head(25), use_container_width=True, hide_index=True)

ui.synthetic_footer()
