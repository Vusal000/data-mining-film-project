"""
Səhifə 1: EV
============

Layihənin vizit kartı: problem, tədqiqat sualı, əsas rəqəmlər və mərhələlər.
"""

import streamlit as st

from src import config
from src import ui

ui.page_setup(
    "🎬 Film Tövsiyə Sistemi",
    "Müştəri reytinqlərinə əsaslanan tövsiyə sisteminin qurulması və "
    "üç yanaşmanın müqayisəli qiymətləndirilməsi",
)

customers, films, ratings = ui.get_data()
split = ui.get_split()
results = ui.get_results()
meta = results.get("meta", {})

# --------------------------------------------------------------------------
# PROBLEM VƏ TƏDQİQAT SUALI
# --------------------------------------------------------------------------

left, right = st.columns([1.15, 1])

with left:
    st.markdown("### 🎯 Problem")
    st.markdown(
        """
        Kataloqda **200 film** var, amma bir müştəri ömrü boyu onların
        yalnız kiçik bir hissəsinə baxır. Müştəri qarşısında uzun bir
        siyahı görəndə nə seçəcəyini bilmir - və çox vaxt heç nə seçmir.

        **Tövsiyə sistemi** bu problemi həll edir: hər müştəriyə məhz onun
        zövqünə uyğun ola biləcək filmləri önə çıxarır.

        Çətinlik ondadır ki, reytinq matrisi demək olar ki, **boşdur** -
        mümkün xanaların yalnız kiçik bir hissəsi doludur. Sistem görmədiyi
        xanaları təxmin etməlidir.
        """
    )

with right:
    st.markdown("### ❓ Tədqiqat sualı")
    ui.card(
        "Əsas sual",
        "Müştəri reytinqlərinə əsaslanan üç fərqli tövsiyə yanaşmasından "
        "hansı daha yaxşı nəticə verir - və <b>“yaxşı” dedikdə nəyi nəzərdə "
        "tuturuq?</b>",
        color=config.COLOR_BURGUNDY,
    )
    ui.card(
        "Alt suallar",
        "1. Bal proqnozunda ən dəqiq model hansıdır? (RMSE)<br>"
        "2. Tövsiyə siyahısında ən çox “tuş gələn” model hansıdır? (Precision@10)<br>"
        "3. Gizli faktorların sayı (k) nəticəyə necə təsir edir?<br>"
        "4. Yeni müştəri üçün (soyuq start) hansı model işləyir?",
        color=config.COLOR_TEAL,
    )

st.divider()

# --------------------------------------------------------------------------
# ƏSAS RƏQƏMLƏR (KPI)
# --------------------------------------------------------------------------

st.markdown("### 📈 Verilənlər bir baxışda")

sparsity = 1 - len(ratings) / (len(customers) * len(films))
possible_cells = len(customers) * len(films)

c1, c2, c3, c4 = st.columns(4)
with c1:
    ui.kpi("Müştəri", f"{len(customers):,}".replace(",", " "), "sintetik profil")
with c2:
    ui.kpi("Film", f"{len(films):,}".replace(",", " "), "1990-2025, 10 janr")
with c3:
    ui.kpi("Reytinq", f"{len(ratings):,}".replace(",", " "), "1-5 ulduz")
with c4:
    ui.kpi(
        "Seyrəklik",
        f"{sparsity:.1%}",
        f"{possible_cells:,}".replace(",", " ") + " xanadan boş olanlar",
    )

ui.explain(
    f"Matrisin <b>{sparsity:.1%}</b>-i boşdur: {len(customers)} müştəri × "
    f"{len(films)} film = {possible_cells:,}".replace(",", " ")
    + f" mümkün xanadan yalnız {len(ratings):,}".replace(",", " ")
    + " ədədi doludur. <b>Məhz bu boşluq tövsiyə sistemini zəruri edir</b> - "
    "sadə ortalama hesablamaqla bu qədər boş xananı doldurmaq mümkün deyil."
)

# --------------------------------------------------------------------------
# BÖLGÜ
# --------------------------------------------------------------------------

b1, b2, b3, b4 = st.columns(4)
with b1:
    ui.kpi("Train", f"{len(split.train):,}".replace(",", " "), "modeli öyrədir")
with b2:
    ui.kpi("Validasiya", f"{len(split.val):,}".replace(",", " "), "yalnız k seçimi")
with b3:
    ui.kpi("Test", f"{len(split.test):,}".replace(",", " "), "yalnız yekun ölçmə")
with b4:
    ui.kpi(
        "Seçilmiş k",
        str(meta.get("best_k", "-")),
        f"{meta.get('svd_epochs', '-')} epoch ayarı üçün",
    )

ui.explain(
    "Bölgü <b>zamana görədir</b>: hər müştərinin ən son 20% reytinqi test "
    "üçün ayrılır. Bu, real vəziyyəti təqlid edir - keçmişi bilirik, "
    "gələcəyi proqnoz edirik. Test seti yalnız <b>bir dəfə</b>, yekun "
    "ölçmədə istifadə olunur."
)

st.divider()

# --------------------------------------------------------------------------
# LAYİHƏNİN MƏRHƏLƏLƏRİ
# --------------------------------------------------------------------------

st.markdown("### 🧩 Layihənin mərhələləri")

STAGES = [
    ("Verilənlərin yaradılması",
     "150 müştəri, 200 film və ~5000 reytinq gizli struktura əsasən "
     "sintetik yaradılır."),
    ("Zamana görə bölgü",
     "Hər müştərinin reytinqləri train / validasiya / test hissələrinə "
     "ayrılır - sızma olmadan."),
    ("Üç modelin qurulması",
     "Populyarlıq (baza), Item-based əməkdaşlıq süzgəci və öz əlimizlə "
     "yazılmış SVD."),
    ("Qiymətləndirmə",
     "RMSE, Precision@10, Recall@10 və kataloq əhatəsi eyni qaydalarla "
     "ölçülür."),
    ("Analiz və təqdimat",
     "k-nın təsiri, izahlı tövsiyələr, soyuq start analizi və bu interaktiv "
     "tətbiq."),
]

columns = st.columns(5)
for index, (column, (title, body)) in enumerate(zip(columns, STAGES), start=1):
    with column:
        st.markdown(
            f'<div class="step"><div class="num">{index}</div>'
            f"<h5>{title}</h5><p>{body}</p></div>",
            unsafe_allow_html=True,
        )

st.write("")
st.divider()

# --------------------------------------------------------------------------
# ÜÇ MODEL - QISA TANIŞLIQ
# --------------------------------------------------------------------------

st.markdown("### 🎞️ Müqayisə olunan üç yanaşma")

m1, m2, m3 = st.columns(3)
with m1:
    ui.card(
        "1. Populyarlıq (baza model)",
        "“Ən çox baxılan filmlər” lövhəsi. Heç kimi tanımır, hamıya eyni "
        "siyahını verir. Digər modellər bundan yaxşı olmalıdır - əks halda "
        "mürəkkəbliyin mənası yoxdur.",
        color=config.MODEL_COLORS["popularity"],
    )
with m2:
    ui.card(
        "2. Əməkdaşlıq süzgəci (Item-CF)",
        "“Bu filmi bəyənənlər həm də bunu bəyənib.” Filmlər arasında "
        "oxşarlığı janrdan yox, <b>insanların qiymətləndirmə davranışından</b> "
        "öyrənir.",
        color=config.MODEL_COLORS["item_cf"],
    )
with m3:
    ui.card(
        "3. SVD (matris faktorizasiyası)",
        "Hər müştəri və filmi <b>gizli xüsusiyyətlər</b> vektoru ilə təsvir "
        "edir. Bu vektorlar SGD alqoritmi ilə sıfırdan öyrədilir - kodu "
        "numpy ilə özümüz yazmışıq.",
        color=config.MODEL_COLORS["svd"],
    )

st.info(
    "👈 Yan paneldən digər səhifələrə keçə bilərsiniz: verilənlərin təhlili, "
    "metodların izahı, modellərin müqayisəsi, fərdi tövsiyələr, soyuq start "
    "təcrübəsi, sistemin arxitekturası və yekun nəticələr.",
    icon="🧭",
)

ui.synthetic_footer()
