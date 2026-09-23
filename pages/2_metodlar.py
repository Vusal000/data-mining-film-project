"""
Səhifə 3: METODLAR
==================

Hər modeli gündəlik bənzətmə + sadə sxem + kiçik düsturla izah edirik.
Qayda: KİÇİK DÜSTUR, BÖYÜK İZAH.
"""

import plotly.graph_objects as go
import streamlit as st

from src import config
from src import ui

ui.page_setup(
    "🧭 Metodlar",
    "Üç yanaşmanın hər biri gündəlik bənzətmə, sadə sxem və kiçik düsturla",
)

results = ui.get_results()
meta = results.get("meta", {})


def diagram(html: str) -> None:
    """Sadə HTML sxemini çərçivədə göstərir."""
    st.markdown(
        f'<div style="background:#FFFFFF;border-radius:13px;padding:1.3rem;'
        f'box-shadow:0 2px 10px rgba(36,26,34,0.08);margin-bottom:0.9rem;">'
        f"{html}</div>",
        unsafe_allow_html=True,
    )


BOX = (
    "display:inline-block;padding:0.55rem 0.9rem;border-radius:10px;"
    "font-size:0.87rem;font-weight:600;text-align:center;line-height:1.35;"
)
ARROW = (
    f'<span style="color:{ui.GOLD};font-size:1.5rem;font-weight:800;'
    'margin:0 0.55rem;vertical-align:middle;">→</span>'
)

# ==========================================================================
# MODEL 1: POPULYARLIQ
# ==========================================================================

st.markdown(
    f'<h3><span class="tag" style="background:{config.MODEL_COLORS["popularity"]}">'
    f"1</span>&nbsp; Populyarlıq - baza model</h3>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.25])

with left:
    ui.card(
        "🎪 Gündəlik bənzətmə",
        "Kinoteatrın girişindəki <b>“Ən çox baxılanlar”</b> lövhəsi. "
        "Lövhə sizi tanımır, adınızı bilmir, nə sevdiyinizi soruşmur - "
        "hamıya eyni siyahını göstərir.<br><br>"
        "Sadədir, amma gözlənildiyindən güclüdür: populyar filmlər "
        "təsadüfən populyar olmur.",
        color=config.MODEL_COLORS["popularity"],
    )
    ui.card(
        "🎯 Niyə bu modelə ehtiyac var?",
        "O, <b>müqayisə xətti</b>dir. Əgər mürəkkəb modellər bu sadə "
        "lövhəni üstələyə bilmirsə, deməli mürəkkəbliyə dəyməz. "
        "Elmi işdə buna <b>baseline</b> deyilir.",
        color=config.COLOR_GOLD,
    )

with right:
    diagram(
        f"""
        <div style="text-align:center">
          <div style="{BOX}background:#F4ECE1;color:{ui.INK};">
            Bütün müştərilər<br><span style="font-size:1.4rem">👤👤👤</span>
          </div>
          {ARROW}
          <div style="{BOX}background:{ui.SLATE};color:#fff;">
            Filmləri reytinq<br>SAYINA görə sırala
          </div>
          {ARROW}
          <div style="{BOX}background:{ui.GOLD};color:#3A2A10;">
            Eyni Top-10<br>siyahısı
          </div>
        </div>
        """
    )
    st.markdown("**Bal proqnozu üçün düstur — “sönümlü orta”:**")
    st.latex(
        r"\hat{r}_i = \frac{\sum r_i + 5 \cdot \mu}{n_i + 5}"
    )
    st.markdown(
        f"""
        - $\\sum r_i$ — filmin aldığı balların cəmi
        - $n_i$ — filmin neçə reytinq aldığı
        - $\\mu$ — bütün reytinqlərin ümumi ortası
        - **5** — “sönümləmə” sabiti (`DAMPING = {config.DAMPING}`)
        """
    )

ui.explain(
    "<b>Niyə adi orta yox, “sönümlü orta”?</b> Təsəvvür edin: bir filmi cəmi "
    "1 nəfər görüb və 5 bal verib. Adi ortası 5.0 olur və o, 500 nəfərin 4.6 "
    "verdiyi filmi üstələyir - bu ədalətsizdir. Sönümlü orta sanki hər filmə "
    "əvvəlcədən ümumi ortada olan <b>5 saxta reytinq</b> əlavə edir. Az "
    "məlumatlı film ümumi ortaya yaxın qalır; çox reytinqi olan film isə "
    "öz həqiqi ortasına çatır."
)

st.divider()

# ==========================================================================
# MODEL 2: ITEM-CF
# ==========================================================================

st.markdown(
    f'<h3><span class="tag" style="background:{config.MODEL_COLORS["item_cf"]}">'
    f"2</span>&nbsp; Əməkdaşlıq süzgəci (Item-based CF)</h3>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.25])

with left:
    ui.card(
        "🧑‍🤝‍🧑 Gündəlik bənzətmə",
        "Kitab mağazasındakı rəf: <b>“Bunu alanlar həm də bunu aldı.”</b><br><br>"
        "Model filmlərin <i>məzmununa</i> baxmır - nə süjetini, nə janrını "
        "oxuyur. O, yalnız <b>insanların davranışına</b> baxır: “bu iki filmə "
        "eyni adamlar oxşar bal veribmi?”",
        color=config.MODEL_COLORS["item_cf"],
    )
    ui.card(
        "🧮 Niyə “düzəldilmiş” (adjusted)?",
        "Kimisi hər filmə 5 verir, kimisi maksimum 3. Əgər xam ballara "
        "baxsaq, səxavətli müştərilər hər şeyi oxşar göstərər. Buna görə "
        "<b>hər müştərinin balından öz ortasını çıxırıq</b>: “bu filmi "
        "öz adi səviyyəsindən yuxarı qiymətləndiribmi?”",
        color=config.COLOR_GOLD,
    )

with right:
    diagram(
        f"""
        <div style="text-align:center">
          <div style="{BOX}background:#F4ECE1;color:{ui.INK};">
            Siz «Səssiz Liman»a<br><b>5 bal</b> verdiniz
          </div>
          {ARROW}
          <div style="{BOX}background:{ui.TEAL};color:#fff;">
            Bu filmə oxşar bal verən<br>digər filmləri tap<br>
            <span style="font-size:0.78rem;opacity:0.85">(30 ən oxşar qonşu)</span>
          </div>
          {ARROW}
          <div style="{BOX}background:{ui.GOLD};color:#3A2A10;">
            «Qarlı Körpü»<br>tövsiyə olunur
          </div>
        </div>
        <div style="text-align:center;margin-top:0.9rem;color:#6B6169;font-size:0.86rem">
          İzah həmişə belə səslənir:
          <i>«Səssiz Liman» filminə yüksək bal verdiyiniz üçün</i>
        </div>
        """
    )
    st.markdown("**Oxşarlıq düsturu (düzəldilmiş kosinus):**")
    st.latex(
        r"sim(i,j) = \frac{\sum_u (r_{ui}-\bar{r}_u)(r_{uj}-\bar{r}_u)}"
        r"{\sqrt{\sum (r_{ui}-\bar{r}_u)^2}\sqrt{\sum (r_{uj}-\bar{r}_u)^2}}"
    )
    st.markdown("**Sıxılma düzəlişi (az ortaq reytinq üçün):**")
    st.latex(
        r"sim_{\text{düzəldilmiş}} = sim(i,j) \cdot \frac{n_{ij}}{n_{ij}+10}"
    )

ui.explain(
    "<b>Sıxılma (shrinkage) nəyə lazımdır?</b> Əgər iki filmi yalnız "
    "<b>2 nəfər</b> qiymətləndiribsə və hər ikisi oxşar bal veribsə, oxşarlıq "
    "1.00 çıxa bilər - amma bu, təsadüf ola bilər. Düstur oxşarlığı ortaq "
    "reytinq sayına görə zəiflədir: 2 ortaq reytinqdə oxşarlıq 2/(2+10) = "
    "<b>0.17</b> dəfəyə düşür, 100 ortaq reytinqdə isə 100/110 = <b>0.91</b>, "
    "yəni demək olar toxunulmadan qalır."
)

st.markdown("**Proqnoz düsturu (30 ən oxşar qonşu ilə):**")
st.latex(
    r"\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{j \in N(i)} sim(i,j)\,(r_{uj}-\bar{r}_u)}"
    r"{\sum_{j \in N(i)} |sim(i,j)|}"
)
ui.explain(
    "Sadə dillə: <b>“Sənin adi səviyyən + oxşar filmlərdə adi səviyyəndən nə "
    "qədər kənara çıxmısansa, onun çəkili ortalaması.”</b> Əgər oxşar filmlərə "
    "adətən öz ortandan yuxarı bal vermisənsə, model bu filmə də yuxarı bal "
    f"proqnozlaşdırır. Qonşuların sayı: <b>{config.N_NEIGHBORS}</b>."
)

st.divider()

# ==========================================================================
# MODEL 3: SVD
# ==========================================================================

st.markdown(
    f'<h3><span class="tag" style="background:{config.MODEL_COLORS["svd"]}">'
    f"3</span>&nbsp; SVD - matris faktorizasiyası (Funk SVD)</h3>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.25])

with left:
    ui.card(
        "🎛️ Gündəlik bənzətmə",
        "Musiqi pultundakı <b>tənzimləyici düymələr</b>. Hər filmin bir neçə "
        "düymədə müəyyən səviyyəsi var (məsələn “nə qədər döyüşlü”, “nə qədər "
        "kədərli”), hər müştərinin isə həmin düymələrə öz marağı var.<br><br>"
        "<b>Əsas incəlik:</b> bu düymələrin adını biz vermirik. Model onları "
        "verilənlərdən <b>özü kəşf edir</b> - buna görə “gizli” (latent) faktor deyilir.",
        color=config.MODEL_COLORS["svd"],
    )
    ui.card(
        "🔁 Necə öyrənir? (SGD)",
        "Hər mövcud reytinqə bir-bir baxır, <b>xəta = həqiqi bal − proqnoz</b> "
        "hesablayır və bütün parametrləri xətanı azaldacaq istiqamətdə "
        "<b>kiçik bir addım</b> dəyişir. Bu proses bütün reytinqlər üzrə "
        "dəfələrlə təkrarlanır - hər təkrara <b>epoch</b> deyilir.",
        color=config.COLOR_GOLD,
    )

with right:
    diagram(
        f"""
        <div style="text-align:center">
          <div style="{BOX}background:#F4ECE1;color:{ui.INK};">
            R<br><span style="font-size:0.78rem">müştəri × film<br>(çox boş)</span>
          </div>
          <span style="color:{ui.BURGUNDY};font-size:1.5rem;font-weight:800;
                margin:0 0.55rem;vertical-align:middle;">≈</span>
          <div style="{BOX}background:{ui.BURGUNDY};color:#fff;">
            P<br><span style="font-size:0.78rem">müştəri × k</span>
          </div>
          <span style="color:{ui.GOLD};font-size:1.4rem;font-weight:800;
                margin:0 0.4rem;vertical-align:middle;">×</span>
          <div style="{BOX}background:{ui.TEAL};color:#fff;">
            Qᵀ<br><span style="font-size:0.78rem">k × film</span>
          </div>
        </div>
        <div style="text-align:center;margin-top:0.9rem;color:#6B6169;font-size:0.86rem">
          Böyük boş matrisi iki kiçik və <b>dolu</b> matrisin hasili kimi yazırıq.<br>
          Hasili hesablayanda <b>boş xanalar da dolur</b> - proqnoz budur.
        </div>
        """
    )
    st.markdown("**Proqnoz düsturu:**")
    st.latex(r"\hat{r}_{ui} = \mu + b_u + b_i + p_u \cdot q_i")
    st.markdown(
        """
        - $\\mu$ — ümumi orta bal
        - $b_u$ — müştərinin meyli (səxavətli / sərt)
        - $b_i$ — filmin meyli (ümumiyyətlə yaxşı / zəif)
        - $p_u \\cdot q_i$ — **gizli zövq uyğunluğu**
        """
    )

st.markdown("**SGD yeniləmə qaydaları:**")
c1, c2 = st.columns(2)
with c1:
    st.latex(r"b_u \leftarrow b_u + lr\,(e_{ui} - reg \cdot b_u)")
    st.latex(r"p_u \leftarrow p_u + lr\,(e_{ui}\,q_i - reg \cdot p_u)")
with c2:
    st.latex(r"b_i \leftarrow b_i + lr\,(e_{ui} - reg \cdot b_i)")
    st.latex(r"q_i \leftarrow q_i + lr\,(e_{ui}\,p_u - reg \cdot q_i)")

st.markdown(
    f"`lr = {config.SVD_LR}` (addım ölçüsü) &nbsp;•&nbsp; "
    f"`reg = {config.SVD_REG}` (requlyarizasiya) &nbsp;•&nbsp; "
    f"`epochs = {config.SVD_EPOCHS}` (tapşırığın tələbi)"
)

ui.explain(
    "<b>Requlyarizasiya (reg) nə edir?</b> Düsturdakı <code>− reg · p</code> "
    "hissəsi parametrləri hər addımda bir az sıfıra doğru çəkir. Bu, modelin "
    "təlim verilənlərini <b>əzbərləməsinin</b> qarşısını alır. Əzbərləyən model "
    "təlimdə əla, yeni verilənlərdə isə pis olur."
)

# --- Təlim əyrisi ---
curve = results.get("svd_training_curve")
if curve is not None:
    fig = go.Figure(
        go.Scatter(
            x=curve["epoch"],
            y=curve["train_rmse"],
            mode="lines+markers",
            line=dict(color=ui.BURGUNDY, width=3),
            marker=dict(size=7),
            hovertemplate="Epoch %{x}<br>Təlim RMSE: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"SVD təlim prosesi (k={meta.get('best_k', '?')})",
        xaxis_title="Epoch (tam dövrə sayı)",
        yaxis_title="Təlim RMSE",
    )
    ui.show(
        fig,
        "Hər epoch-dan sonra modelin öz təlim verilənlərindəki xətası azalır - "
        "yəni alqoritm həqiqətən öyrənir. <b>Amma diqqət:</b> bu, təlim "
        "xətasıdır. Modelin yeni verilənlərdə nə qədər yaxşı olduğunu "
        "“Model müqayisəsi” səhifəsindəki validasiya əyrisi göstərir.",
        height=340,
        showlegend=False,
    )

st.divider()

# ==========================================================================
# YANAŞMALARIN MÜQAYİSƏSİ
# ==========================================================================

st.markdown("### 🔍 Üç yanaşma yan-yana")

comparison_html = f"""
<table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
  <tr style="background:{ui.BURGUNDY};color:#fff;">
    <th style="padding:0.65rem;text-align:left;">Sual</th>
    <th style="padding:0.65rem;">Populyarlıq</th>
    <th style="padding:0.65rem;">Item-CF</th>
    <th style="padding:0.65rem;">SVD</th>
  </tr>
  <tr style="background:#FFFFFF;">
    <td style="padding:0.6rem;font-weight:600;">Müştərini tanıyırmı?</td>
    <td style="padding:0.6rem;text-align:center;">❌ Xeyr</td>
    <td style="padding:0.6rem;text-align:center;">✅ Bəli</td>
    <td style="padding:0.6rem;text-align:center;">✅ Bəli</td>
  </tr>
  <tr style="background:#FAF6F0;">
    <td style="padding:0.6rem;font-weight:600;">Nəyə əsaslanır?</td>
    <td style="padding:0.6rem;text-align:center;">Reytinq sayına</td>
    <td style="padding:0.6rem;text-align:center;">Film-film oxşarlığına</td>
    <td style="padding:0.6rem;text-align:center;">Gizli vektorlara</td>
  </tr>
  <tr style="background:#FFFFFF;">
    <td style="padding:0.6rem;font-weight:600;">İzah etmək asandırmı?</td>
    <td style="padding:0.6rem;text-align:center;">✅ Çox asan</td>
    <td style="padding:0.6rem;text-align:center;">✅ Asan</td>
    <td style="padding:0.6rem;text-align:center;">⚠️ Çətin</td>
  </tr>
  <tr style="background:#FAF6F0;">
    <td style="padding:0.6rem;font-weight:600;">Yeni müştəri (soyuq start)</td>
    <td style="padding:0.6rem;text-align:center;">✅ İşləyir</td>
    <td style="padding:0.6rem;text-align:center;">❌ Zəif</td>
    <td style="padding:0.6rem;text-align:center;">⚠️ Fold-in lazımdır</td>
  </tr>
  <tr style="background:#FFFFFF;">
    <td style="padding:0.6rem;font-weight:600;">Öyrədilən parametr sayı</td>
    <td style="padding:0.6rem;text-align:center;">Yoxdur</td>
    <td style="padding:0.6rem;text-align:center;">Yoxdur (hesablanır)</td>
    <td style="padding:0.6rem;text-align:center;">(150+200)×k + meyllər</td>
  </tr>
</table>
"""
st.markdown(comparison_html, unsafe_allow_html=True)

st.write("")
ui.explain(
    "Heç bir model hər sütunda qalib deyil. Populyarlıq sadə və soyuq startda "
    "etibarlıdır, Item-CF izahlıdır və kataloqu geniş əhatə edir, SVD isə ən "
    "güclü riyazi modeldir - amma <b>izah etmək və yeni istifadəçiyə tətbiq "
    "etmək daha çətindir</b>. “Model müqayisəsi” səhifəsində bunun rəqəmlərlə "
    "necə göründüyünü görəcəksiniz."
)

ui.synthetic_footer()
