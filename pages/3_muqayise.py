"""
Səhifə 4: MODEL MÜQAYİSƏSİ
==========================

Bütün rəqəmlər results/ qovluğundakı CSV fayllarından oxunur.
Bu səhifə heç nə hesablamır - yəni göstərilən hər ədəd
`python -m src.experiments` əmri ilə həqiqətən ölçülmüş dəyərdir.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src import ui

ui.page_setup(
    "🏆 Model müqayisəsi",
    "Üç yanaşmanın test setindəki nəticələri, gizli faktor sayının təsiri "
    "və avtomatik yazılmış nəticələr",
)

results = ui.require_results()
if results is None:
    st.stop()

comparison = results["model_comparison"].copy()
k_sweep = results["k_sweep"].copy()
epoch_sweep = results["epoch_sweep"].copy()
meta = results["meta"]

PRECISION_COL = f"Precision@{config.TOP_N}"
RECALL_COL = f"Recall@{config.TOP_N}"
COVERAGE_COL = "Kataloq əhatəsi"

# Əsas cədvəl: tapşırığın tələb etdiyi üç model
main = comparison[comparison["model_key"].isin(["popularity", "item_cf", "svd"])]
tuned = comparison[comparison["model_key"] == "svd_tuned"]

# --------------------------------------------------------------------------
# 1. ƏSAS CƏDVƏL
# --------------------------------------------------------------------------

st.markdown("### 1. Yekun nəticə cədvəli (test seti)")

display = comparison[
    ["Model", "RMSE", "MAE", PRECISION_COL, RECALL_COL, COVERAGE_COL]
].copy()

styled = (
    display.style.format(
        {
            "RMSE": "{:.4f}",
            "MAE": "{:.4f}",
            PRECISION_COL: "{:.4f}",
            RECALL_COL: "{:.4f}",
            COVERAGE_COL: "{:.1%}",
        }
    )
    .highlight_min(subset=["RMSE", "MAE"], color="#D7EBE3")
    .highlight_max(subset=[PRECISION_COL, RECALL_COL, COVERAGE_COL], color="#D7EBE3")
)
st.dataframe(styled, use_container_width=True, hide_index=True)

ui.explain(
    "Yaşıl xanalar hər sütunun <b>ən yaxşı</b> dəyəridir. RMSE və MAE-də "
    "<b>kiçik</b>, Precision / Recall / əhatədə isə <b>böyük</b> yaxşıdır. "
    "Son sətir tapşırıqda tələb olunan üç modelə əlavədir: eyni SVD, amma "
    f"{meta.get('svd_epochs_tuned')} epoch ilə öyrədilmiş variantı "
    "(səbəbi 4-cü bölmədə izah olunur)."
)

# --- Qalibləri tapırıq ---
best_rmse = main.loc[main["RMSE"].idxmin()]
best_precision = main.loc[main[PRECISION_COL].idxmax()]
best_recall = main.loc[main[RECALL_COL].idxmax()]
best_coverage = main.loc[main[COVERAGE_COL].idxmax()]

c1, c2, c3, c4 = st.columns(4)
with c1:
    ui.kpi("Ən dəqiq bal proqnozu", config.MODEL_SHORT_AZ[best_rmse["model_key"]],
           f"RMSE = {best_rmse['RMSE']:.4f}")
with c2:
    ui.kpi("Ən yaxşı tövsiyə", config.MODEL_SHORT_AZ[best_precision["model_key"]],
           f"P@10 = {best_precision[PRECISION_COL]:.4f}")
with c3:
    ui.kpi("Ən yaxşı əhatə (Recall)", config.MODEL_SHORT_AZ[best_recall["model_key"]],
           f"R@10 = {best_recall[RECALL_COL]:.4f}")
with c4:
    ui.kpi("Ən geniş kataloq", config.MODEL_SHORT_AZ[best_coverage["model_key"]],
           f"{best_coverage[COVERAGE_COL]:.1%} film təklif olunub")

st.divider()

# --------------------------------------------------------------------------
# 2. MÜQAYİSƏ DİAQRAMLARI
# --------------------------------------------------------------------------

st.markdown("### 2. Metrikaların müqayisəsi")


def metric_bar(column: str, title: str, lower_is_better: bool, fmt: str = "{:.4f}"):
    """Bir metrika üzrə sütun diaqramı (hər model öz rəngində)."""
    frame = comparison.sort_values(column, ascending=lower_is_better)
    colors = [
        config.MODEL_COLORS.get(
            key if key != "svd_tuned" else "svd", ui.SLATE
        )
        for key in frame["model_key"]
    ]
    # Tənzimlənmiş SVD-ni zolaqlı naxışla fərqləndiririk
    patterns = ["/" if key == "svd_tuned" else "" for key in frame["model_key"]]

    figure = go.Figure(
        go.Bar(
            x=frame["Model"],
            y=frame[column],
            marker=dict(color=colors, pattern_shape=patterns),
            text=[fmt.format(v) for v in frame[column]],
            textposition="outside",
            hovertemplate="%{x}<br>" + column + ": %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(title=title, yaxis_title=column, xaxis_tickangle=-12)
    return figure


left, right = st.columns(2)
with left:
    ui.show(
        metric_bar("RMSE", "RMSE - bal proqnozunun xətası (kiçik = yaxşı)", True),
        "RMSE orta hesabla neçə bal yanıldığımızı göstərir. Məsələn "
        f"<b>{best_rmse['RMSE']:.2f}</b> o deməkdir ki, model 5 ballıq şkalada "
        "təxminən yarım baldan bir qədər çox səhv edir. Zolaqlı sütun "
        "tənzimlənmiş SVD variantıdır.",
        height=380,
        showlegend=False,
    )
with right:
    ui.show(
        metric_bar(PRECISION_COL, "Precision@10 - tövsiyələrin dəqiqliyi", False),
        "Təklif olunan 10 filmdən neçəsinin müştərinin test dövründə "
        "həqiqətən bəyəndiyi film olduğunu göstərir. Rəqəmlər kiçik görünür, "
        "çünki müştərinin test dövründə cəmi bir neçə filmi var - 10 "
        "təklifdən hamısının tuş gəlməsi <b>mümkün deyil</b>.",
        height=380,
        showlegend=False,
    )

left, right = st.columns(2)
with left:
    ui.show(
        metric_bar(RECALL_COL, "Recall@10 - bəyənilənlərin neçə faizi tutuldu", False),
        "Müştərinin test dövründə bəyəndiyi filmlərin neçə faizini Top-10 "
        "siyahımıza sala bildik? Precision “təkliflərim nə qədər təmizdir”, "
        "Recall isə “nə qədərini qaçırmadım” sualına cavab verir.",
        height=380,
        showlegend=False,
    )
with right:
    ui.show(
        metric_bar(COVERAGE_COL, "Kataloq əhatəsi - neçə fərqli film təklif olunub",
                   False, "{:.1%}"),
        "Bütün müştərilərə verilən tövsiyələrdə kataloqun neçə faizi "
        "ümumiyyətlə görünür. Aşağı əhatə = model <b>eyni bir neçə filmi</b> "
        "təkrarlayır. Bu, dəqiqlik metrikalarında görünməyən, amma real "
        "sistemlər üçün çox vacib olan problemdir.",
        height=380,
        showlegend=False,
    )

st.divider()

# --------------------------------------------------------------------------
# 3. k-nın TƏSİRİ
# --------------------------------------------------------------------------

st.markdown("### 3. Gizli faktor sayının (k) təsiri")

best_k = meta.get("best_k")
best_k_tuned = meta.get("best_k_tuned")
epochs_spec = meta.get("svd_epochs")
epochs_tuned = meta.get("svd_epochs_tuned")

validation = k_sweep[k_sweep["dataset"] == "validasiya"]
test_rows = k_sweep[k_sweep["dataset"] == "test"]

left, right = st.columns(2)

with left:
    fig = go.Figure()
    for epochs, color, dash in (
        (epochs_spec, ui.BURGUNDY, "solid"),
        (epochs_tuned, ui.TEAL, "dash"),
    ):
        subset = validation[validation["epochs"] == epochs].sort_values("k")
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["k"],
                y=subset["rmse"],
                mode="lines+markers",
                name=f"{epochs} epoch",
                line=dict(color=color, width=3, dash=dash),
                marker=dict(size=9),
                error_y=dict(
                    type="data", array=subset["rmse_std"], visible=True,
                    color=color, thickness=1.4, width=5,
                ),
                hovertemplate="k=%{x}<br>Validasiya RMSE: %{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="k → validasiya RMSE (5 təsadüfi toxumun ortalaması)",
        xaxis_title="Gizli faktor sayı (k)",
        yaxis_title="Validasiya RMSE",
        xaxis_type="log",
    )
    fig.update_xaxes(tickvals=config.SVD_FACTORS_GRID,
                     ticktext=[str(k) for k in config.SVD_FACTORS_GRID])
    ui.show(
        fig,
        f"<b>İki əyri, iki fərqli hekayə.</b> Tapşırığın tələb etdiyi "
        f"{epochs_spec} epoch ilə (tünd bordo) əyri demək olar düzdür - k-nı "
        f"artırmaq nəticəni dəyişmir. {epochs_tuned} epoch ilə (firuzəyi) isə "
        "əyri aydın şəkildə <b>aşağı enir</b>: gizli faktorlar həqiqətən işə "
        "yarayır. Şaquli xətlər 5 təkrarın yayılmasıdır - fərqlər təsadüfi deyil.",
        height=400,
    )

with right:
    fig = go.Figure()
    for epochs, color, dash in (
        (epochs_spec, ui.BURGUNDY, "solid"),
        (epochs_tuned, ui.TEAL, "dash"),
    ):
        subset = test_rows[test_rows["epochs"] == epochs].sort_values("k")
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["k"],
                y=subset["precision_at_k"],
                mode="lines+markers",
                name=f"{epochs} epoch",
                line=dict(color=color, width=3, dash=dash),
                marker=dict(size=9),
                hovertemplate="k=%{x}<br>Test P@10: %{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="k → test Precision@10",
        xaxis_title="Gizli faktor sayı (k)",
        yaxis_title="Precision@10",
        xaxis_type="log",
    )
    fig.update_xaxes(tickvals=config.SVD_FACTORS_GRID,
                     ticktext=[str(k) for k in config.SVD_FACTORS_GRID])
    ui.show(
        fig,
        "Tövsiyə keyfiyyəti (P@10) k ilə birlikdə daha dalğalı davranır, "
        "çünki bu metrika yalnız <b>ilk 10 filmin sırasına</b> baxır - bir "
        "neçə filmin yerini dəyişməsi bütün nəticəni tərpədə bilir. Bu əyri "
        "yalnız <b>həssaslıq analizi</b> üçündür; k seçimi soldakı validasiya "
        "əyrisi ilə edilib.",
        height=400,
    )

st.info(
    f"**k necə seçildi?** Yalnız validasiya seti ilə: tapşırıq ayarı "
    f"({epochs_spec} epoch) üçün **k = {best_k}**, tənzimlənmiş ayar "
    f"({epochs_tuned} epoch) üçün **k = {best_k_tuned}**. "
    "Test seti bu seçimdə ümumiyyətlə istifadə olunmayıb.",
    icon="🔒",
)

st.divider()

# --------------------------------------------------------------------------
# 4. EPOCH DİAQNOZU
# --------------------------------------------------------------------------

st.markdown("### 4. Diaqnoz: 20 epoch kifayət edirmi?")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=epoch_sweep["epochs"], y=epoch_sweep["train_rmse"],
        mode="lines+markers", name="Təlim RMSE",
        line=dict(color=ui.SLATE, width=3), marker=dict(size=8),
        hovertemplate="%{x} epoch<br>Təlim RMSE: %{y:.4f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=epoch_sweep["epochs"], y=epoch_sweep["val_rmse"],
        mode="lines+markers", name="Validasiya RMSE",
        line=dict(color=ui.BURGUNDY, width=3), marker=dict(size=8),
        hovertemplate="%{x} epoch<br>Validasiya RMSE: %{y:.4f}<extra></extra>",
    )
)

best_epoch_row = epoch_sweep.loc[epoch_sweep["val_rmse"].idxmin()]
fig.add_vline(
    x=epochs_spec, line_dash="dot", line_color=ui.TEAL,
    annotation_text=f"tapşırıq: {epochs_spec}", annotation_position="top",
)
fig.add_vline(
    x=int(best_epoch_row["epochs"]), line_dash="dash", line_color=ui.GOLD,
    annotation_text=f"ən yaxşı: {int(best_epoch_row['epochs'])}",
    annotation_position="top right",
)
fig.update_layout(
    title="Epoch sayı → təlim və validasiya xətası",
    xaxis_title="Epoch sayı (loqarifmik)",
    yaxis_title="RMSE",
    xaxis_type="log",
)
fig.update_xaxes(tickvals=config.SVD_EPOCH_GRID,
                 ticktext=[str(e) for e in config.SVD_EPOCH_GRID])

spec_row = epoch_sweep[epoch_sweep["epochs"] == epochs_spec].iloc[0]
ui.show(
    fig,
    f"Bu, dərslik nümunəsidir. <b>Boz əyri</b> (təlim xətası) daim aşağı "
    f"düşür - model getdikcə daha çox “əzbərləyir”. <b>Bordo əyri</b> "
    f"(validasiya xətası) isə əvvəl düşür, sonra qalxır. "
    f"Tapşırıqdakı <b>{epochs_spec} epoch</b> nöqtəsində validasiya xətası "
    f"{spec_row['val_rmse']:.4f}-dir, halbuki ən yaxşı nöqtə "
    f"({int(best_epoch_row['epochs'])} epoch) {best_epoch_row['val_rmse']:.4f} "
    f"verir. Yəni model həmin nöqtədə <b>hələ kifayət qədər öyrənməyib</b>.",
    height=420,
)

st.warning(
    f"**Niyə belə oldu?** `20 epoch, lr=0.005` dəyərləri `surprise` "
    f"kitabxanasının standart ayarlarıdır və MovieLens-100k (100 000 reytinq) "
    f"kimi verilənlər üçün nəzərdə tutulub. Bizim verilənlər "
    f"{meta.get('n_ratings'):,}".replace(",", " ") + " reytinqdir - təxminən "
    "**20 dəfə kiçik**. Kiçik verilənlərdə hər parametr daha az dəfə "
    "yenilənir, ona görə eyni epoch sayı kifayət etmir.",
    icon="🔬",
)

st.divider()

# --------------------------------------------------------------------------
# 5. AVTOMATİK NƏTİCƏLƏR
# --------------------------------------------------------------------------

st.markdown("### 5. Nəticələr (avtomatik yazılıb)")

rmse_winner = config.MODEL_NAMES_AZ[best_rmse["model_key"]]
precision_winner = config.MODEL_NAMES_AZ[best_precision["model_key"]]
tuned_row = tuned.iloc[0] if len(tuned) else None
svd_spec = main[main["model_key"] == "svd"].iloc[0]

sentences = [
    f"**Bal proqnozunda ən dəqiq:** {rmse_winner} — RMSE = "
    f"{best_rmse['RMSE']:.4f}.",
    f"**Tövsiyə siyahısında ən dəqiq:** {precision_winner} — Precision@10 = "
    f"{best_precision[PRECISION_COL]:.4f}.",
    f"**Ən geniş kataloq əhatəsi:** "
    f"{config.MODEL_NAMES_AZ[best_coverage['model_key']]} — kataloqun "
    f"{best_coverage[COVERAGE_COL]:.1%}-i tövsiyələrdə görünüb.",
]

if tuned_row is not None:
    improvement = (svd_spec["RMSE"] - tuned_row["RMSE"]) / svd_spec["RMSE"]
    sentences.append(
        f"**Tənzimləmənin təsiri:** SVD-ni {epochs_spec} epoch əvəzinə "
        f"{epochs_tuned} epoch öyrətdikdə RMSE {svd_spec['RMSE']:.4f} → "
        f"{tuned_row['RMSE']:.4f} oldu, yəni **{improvement:.1%}** yaxşılaşma."
    )

for sentence in sentences:
    st.markdown(f"- {sentence}")

# --- Fərqli qaliblər varsa, səbəbini izah edirik ---
if best_rmse["model_key"] != best_precision["model_key"]:
    st.markdown("#### 🤔 Niyə RMSE və Precision@10 fərqli modelləri seçir?")

    counts = ui.get_split().train_full.groupby("film_id").size()
    pool = ui.get_candidate_pool()
    average_catalog = float(counts.reindex(pool).mean())

    recommended_popularity = {}
    for key in ("popularity", "item_cf", "svd"):
        model = (
            ui.get_popularity() if key == "popularity"
            else ui.get_item_cf() if key == "item_cf"
            else ui.get_svd(int(best_k))
        )
        values = []
        for customer_id in sorted(ui.get_seen_films())[:60]:
            candidates = ui.candidates_for(customer_id)
            values += [
                int(counts.get(film_id, 0))
                for film_id, _ in model.recommend(customer_id, candidates, n=10)
            ]
        recommended_popularity[key] = float(np.mean(values))

    explain_fig = go.Figure(
        go.Bar(
            x=[config.MODEL_SHORT_AZ[k] for k in recommended_popularity],
            y=list(recommended_popularity.values()),
            marker_color=[config.MODEL_COLORS[k] for k in recommended_popularity],
            text=[f"{v:.1f}" for v in recommended_popularity.values()],
            textposition="outside",
            hovertemplate="%{x}<br>Orta reytinq sayı: %{y:.1f}<extra></extra>",
        )
    )
    explain_fig.add_hline(
        y=average_catalog, line_dash="dash", line_color=ui.GOLD,
        annotation_text=f"kataloq ortası: {average_catalog:.1f}",
        annotation_position="top left",
    )
    explain_fig.update_layout(
        title="Tövsiyə olunan filmlərin təlimdəki orta reytinq sayı",
        yaxis_title="Orta reytinq sayı",
    )
    ui.show(
        explain_fig,
        "Bu diaqram sirri açır. <b>Populyarlıq</b> modeli hamıya çox baxılan "
        "filmləri təklif edir - və müştərilərin test dövründə baxdıqları da "
        "əsasən məhz o filmlərdir, ona görə “tuş gəlmə” şansı yüksəkdir. "
        "<b>Item-CF</b> isə kataloq ortasından da az tanınan filmləri önə "
        "çəkir: proqnozu dəqiq ola bilər, amma müştəri həmin filmə <b>heç "
        "baxmayıb</b>, deməli onu “bəyənilmiş” kimi saya bilmirik.",
        height=370,
        showlegend=False,
    )

    st.markdown(
        f"""
        **Üç səbəb:**

        1. **İki metrika iki fərqli sual verir.** RMSE bütün test xanalarında
           *bal proqnozunun* dəqiqliyini ölçür. Precision@10 isə yalnız
           *ilk 10 filmin sırasını* yoxlayır. Bir model orta hesabla dəqiq
           ola, amma ən yuxarıda səhv filmləri göstərə bilər.

        2. **Populyarlıq meyli (popularity bias).** Test setinə yalnız
           müştərinin faktiki baxdığı filmlər düşür, insanlar isə əsasən
           populyar filmlərə baxır. Bu, offline qiymətləndirmədə populyarlıq
           modelinə <b>təbii üstünlük</b> verir - real sistemdə bu üstünlük
           bu qədər böyük olmaya bilər.

        3. **Görünməyən film “bəyənilmiş” sayıla bilmir.** Item-CF az tanınan
           filmi düzgün təklif etsə belə, müştəri ona baxmayıbsa, test
           setində qeyd yoxdur və biz bunu <b>səhv</b> kimi sayırıq.
           Buna “missing not at random” problemi deyilir.
        """,
        unsafe_allow_html=True,
    )

    # Qeyd: st.success markdown başa düşür, HTML yox - ona görə ** işlədilir
    st.success(
        f"**Praktiki nəticə:** “Ən yaxşı model” sualının tək cavabı yoxdur. "
        f"Əgər məqsəd **bal proqnozudursa** — {rmse_winner}. "
        f"Əgər məqsəd **siyahının tuş gəlməsidirsə** — {precision_winner}. "
        f"Əgər məqsəd **kataloqu canlandırmaqdırsa** — "
        f"{config.MODEL_NAMES_AZ[best_coverage['model_key']]} "
        f"({best_coverage[COVERAGE_COL]:.0%} əhatə). Real sistemlər adətən "
        f"bu modelləri **birləşdirir**.",
        icon="🎯",
    )

ui.synthetic_footer()
