"""
Səhifə 7: ARXİTEKTURA
=====================

Sistemin necə işlədiyini proqramlaşdırma bilməyən adama izah edir:
verilənlərin axını, qovluq quruluşu və tətbiq açılanda nə baş verir.
"""

import streamlit as st

from src import config
from src import ui

ui.page_setup(
    "🏗️ Sistemin arxitekturası",
    "Verilənlərin axını, qovluq quruluşu və tətbiqin işə düşmə ardıcıllığı",
)

results = ui.get_results()
meta = results.get("meta", {})

# --------------------------------------------------------------------------
# 1. VERİLƏNLƏRİN AXINI
# --------------------------------------------------------------------------

st.markdown("### 1. Verilənlərin axını")

STEPS = [
    ("1", "Yaradılma", "generate_data.py", ui.SLATE,
     "Gizli struktura əsasən 150 müştəri, 200 film və ~5000 reytinq yaradılır "
     "və üç CSV faylına yazılır."),
    ("2", "Bölgü", "split.py", ui.TEAL,
     "Hər müştərinin reytinqləri zamana görə sıralanır: son 20% test, "
     "qalanın son 10%-i validasiya."),
    ("3", "Modellər", "models/", config.COLOR_GOLD,
     "Üç model yalnız təlim verilənləri ilə öyrədilir: populyarlıq, "
     "Item-CF və SVD."),
    ("4", "Qiymətləndirmə", "evaluation.py", "#A8452F",
     "RMSE, Precision@10, Recall@10 və kataloq əhatəsi eyni qaydalarla "
     "hesablanır."),
    ("5", "Nəticələr", "results/*.csv", ui.BURGUNDY,
     "Bütün rəqəmlər CSV fayllarına yazılır - tətbiq onları yalnız oxuyur."),
    ("6", "Dashboard", "app.py + pages/", "#5A1530",
     "Streamlit tətbiqi hazır rəqəmləri interaktiv diaqramlarda göstərir."),
]

flow_html = '<div style="display:flex;align-items:stretch;gap:0.35rem;flex-wrap:wrap;">'
for index, (number, title, file_name, color, description) in enumerate(STEPS):
    flow_html += f"""
    <div style="flex:1;min-width:150px;background:{color};color:#fff;
                border-radius:12px;padding:0.85rem 0.8rem;">
      <div style="font-size:0.72rem;opacity:0.8;font-weight:700;">ADDIM {number}</div>
      <div style="font-size:1.0rem;font-weight:800;margin:0.2rem 0;">{title}</div>
      <div style="font-family:monospace;font-size:0.74rem;opacity:0.9;
                  background:rgba(255,255,255,0.16);border-radius:5px;
                  padding:0.12rem 0.35rem;display:inline-block;">{file_name}</div>
      <div style="font-size:0.79rem;margin-top:0.45rem;line-height:1.45;
                  opacity:0.95;">{description}</div>
    </div>"""
    if index < len(STEPS) - 1:
        flow_html += (
            f'<div style="display:flex;align-items:center;color:{ui.GOLD};'
            f'font-size:1.5rem;font-weight:800;">→</div>'
        )
flow_html += "</div>"

st.markdown(flow_html, unsafe_allow_html=True)
st.write("")

ui.explain(
    "Axın <b>bir istiqamətlidir</b>: sol tərəf sağ tərəfi qidalandırır, geri "
    "əlaqə yoxdur. Bunun ən vacib nəticəsi budur - <b>tətbiq heç bir metrika "
    "hesablamır</b>. O, yalnız 5-ci addımda yazılmış CSV faylarını oxuyur. "
    "Beləliklə ekranda gördüyünüz hər rəqəm əvvəlcədən, nəzarətli şəraitdə "
    "hesablanmış həqiqi ölçmədir."
)

st.divider()

# --------------------------------------------------------------------------
# 2. QOVLUQ QURULUŞU
# --------------------------------------------------------------------------

st.markdown("### 2. Qovluq quruluşu — hər faylın bir işi var")

STRUCTURE = [
    ("📁 data/", "", "Verilənlər qovluğu", True),
    ("　📄 generate_data.py", "Sintetik verilənləri yaradır",
     "Gizli keyfiyyət, populyarlıq və janr zövqü əsasında CSV faylları qurur.", False),
    ("　📊 customers.csv", "150 müştəri", "id, ad, yaş qrupu, şəhər, qeydiyyat tarixi", False),
    ("　📊 films.csv", "200 film", "id, ad, il, janr(lar)", False),
    ("　📊 ratings.csv", "~5000 reytinq", "müştəri, film, bal (1-5), zaman", False),

    ("📁 src/", "", "Analitik nüvə", True),
    ("　📄 config.py", "Bütün parametrlər bir yerdə",
     "Toxum, rənglər, model ayarları, qiymətləndirmə hədləri.", False),
    ("　📄 data_loader.py", "CSV oxuyur",
     "Faylları yükləyir və id ↔ matris indeksi körpüsünü qurur.", False),
    ("　📄 split.py", "Zamana görə bölgü",
     "train / validasiya / test ayırır və sızma yoxlaması funksiyası verir.", False),
    ("　📄 evaluation.py", "Metrikalar",
     "RMSE, MAE, Precision@10, Recall@10, kataloq əhatəsi.", False),
    ("　📄 experiments.py", "Bütün eksperimentlər",
     "k seçimi, epoch diaqnozu, yekun müqayisə, soyuq start → results/", False),
    ("　📄 ui.py", "Tətbiqin ümumi dizaynı",
     "Rənglər, CSS, izah blokları və keşlənmiş verilən/model funksiyaları.", False),

    ("📁 src/models/", "", "Üç model", True),
    ("　📄 base.py", "Ümumi interfeys",
     "fit / predict / recommend / explain - hər üç model eyni qaydaya tabedir.", False),
    ("　📄 popularity.py", "Baza model", "Reytinq sayı + sönümlü orta.", False),
    ("　📄 item_cf.py", "Əməkdaşlıq süzgəci",
     "Düzəldilmiş kosinus + sıxılma + 30 qonşu.", False),
    ("　📄 svd.py", "Matris faktorizasiyası",
     "SGD ilə öyrədilən meyilli SVD və yeni istifadəçi üçün fold-in.", False),

    ("📁 tests/", "", "Testlər", True),
    ("　📄 test_split.py", "Bölgü düzgündürmü?",
     "Dəstlər kəsişmir, test zamanca sonuncudur, bölgü təkrarlanandır.", False),
    ("　📄 test_no_leakage_in_models.py", "Model testi görübmü?",
     "Hər modelin daxilinə baxıb test reytinqlərinin olmadığını təsdiqləyir.", False),
    ("　📄 test_models_behaviour.py", "Davranış testləri",
     "İzahların doğruluğu, metrika aralıqları, fold-in işləkliyi.", False),

    ("📁 pages/", "", "Tətbiqin səhifələri", True),
    ("　📄 0_ev.py … 7_neticeler.py", "8 səhifə",
     "Ev, Verilənlər, Metodlar, Müqayisə, Tövsiyələr, Soyuq start, "
     "Arxitektura, Nəticələr.", False),

    ("📁 results/", "", "Hesablanmış nəticələr", True),
    ("　📊 model_comparison.csv", "Yekun cədvəl", "Üç modelin test metrikaları.", False),
    ("　📊 k_sweep.csv", "k eksperimenti", "Hər k üçün validasiya və test nəticələri.", False),
    ("　📊 epoch_sweep.csv", "Epoch diaqnozu", "Təlim və validasiya RMSE əyriləri.", False),
    ("　📊 cold_start.csv", "Soyuq start", "3/5/10 reytinq üçün Precision@10.", False),
    ("　📄 meta.json", "Təkrarlanabilirlik", "Toxum, seçilmiş k, bütün parametrlər.", False),

    ("📄 app.py", "Giriş nöqtəsi", "Səhifə menyusunu qurur.", True),
    ("📄 conftest.py", "pytest hazırlığı", "Testlər üçün ümumi verilənlər.", True),
    ("📄 requirements.txt", "Paket siyahısı", "Dəqiq versiyalarla.", True),
    ("📄 README.md", "Sənədləşmə", "Quraşdırma, işə salma, qaydalar.", True),
]

rows = ""
for name, short, description, is_header in STRUCTURE:
    background = "#F4ECE1" if is_header else "#FFFFFF"
    weight = "800" if is_header else "600"
    color = ui.BURGUNDY if is_header else ui.INK
    rows += f"""
    <tr style="background:{background};">
      <td style="padding:0.42rem 0.7rem;font-family:monospace;font-size:0.84rem;
                 font-weight:{weight};color:{color};white-space:nowrap;">{name}</td>
      <td style="padding:0.42rem 0.7rem;font-size:0.84rem;font-weight:700;
                 color:{ui.TEAL};white-space:nowrap;">{short}</td>
      <td style="padding:0.42rem 0.7rem;font-size:0.84rem;color:#564B52;">{description}</td>
    </tr>"""

st.markdown(
    f'<table style="width:100%;border-collapse:collapse;border-radius:10px;'
    f'overflow:hidden;box-shadow:0 2px 10px rgba(36,26,34,0.08);">{rows}</table>',
    unsafe_allow_html=True,
)

st.write("")
ui.explain(
    "Hər faylın <b>yalnız bir işi</b> var. Bu, təsadüfi deyil: bölgü kodunu "
    "dəyişəndə modellərə toxunmaq lazım gəlmir, model əlavə edəndə "
    "qiymətləndirmə kodu dəyişmir. Proqramlaşdırmada buna <b>“məsuliyyətlərin "
    "ayrılması”</b> deyilir və kodu başa düşülən saxlayan əsas prinsipdir."
)

st.divider()

# --------------------------------------------------------------------------
# 3. TƏTBİQ AÇILANDA NƏ BAŞ VERİR?
# --------------------------------------------------------------------------

st.markdown("### 3. `streamlit run app.py` yazanda nə baş verir?")

STARTUP = [
    ("Streamlit serveri qalxır",
     "Kompüterinizdə kiçik bir veb-server işə düşür və brauzer açılır. "
     "İnternetə çıxış lazım deyil - hər şey lokal işləyir."),
    ("app.py oxunur",
     "Səhifə ayarları qurulur və 8 səhifə yan panel menyusuna yazılır. "
     "Bu mərhələdə heç bir hesablama olmur."),
    ("Seçilmiş səhifə işə düşür",
     "Yalnız açdığınız səhifənin kodu yerinə yetirilir. Digər səhifələr "
     "toxunulmamış qalır - buna görə açılış sürətlidir."),
    ("Verilənlər oxunur (bir dəfə)",
     "ui.get_data() üç CSV faylını oxuyur. @st.cache_data sayəsində bu, "
     "yalnız birinci dəfə baş verir; sonra yaddaşdan götürülür."),
    ("Bölgü hesablanır (bir dəfə)",
     "ui.get_split() zamana görə bölgünü qurur və keşləyir."),
    ("Modellər öyrədilir (lazım olduqda)",
     "Yalnız model tələb edən səhifələrdə. @st.cache_resource hər (k, epoch) "
     "cütü üçün modeli bir dəfə öyrədir - slayderi geri qaytardıqda "
     "yenidən öyrətmə olmur."),
    ("Nəticələr oxunur",
     "results/ qovluğundakı CSV faylları yüklənir. Bütün metrikalar buradan "
     "gəlir - tətbiq onları özü hesablamır."),
    ("Diaqramlar çəkilir",
     "Plotly interaktiv diaqramları brauzerdə qurulur: yaxınlaşdırmaq, "
     "üzərinə gəlmək və sıra gizlətmək mümkündür."),
]

columns = st.columns(2)
for index, (title, description) in enumerate(STARTUP):
    with columns[index % 2]:
        st.markdown(
            f'<div style="display:flex;gap:0.7rem;background:#fff;'
            f'border-radius:11px;padding:0.75rem 0.9rem;margin-bottom:0.55rem;'
            f'box-shadow:0 2px 8px rgba(36,26,34,0.07);">'
            f'<div style="flex:0 0 28px;height:28px;border-radius:50%;'
            f'background:{ui.GOLD};color:#3A2A10;font-weight:800;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:0.85rem;">{index + 1}</div>'
            f'<div><div style="font-weight:700;color:{ui.BURGUNDY};'
            f'font-size:0.93rem;">{title}</div>'
            f'<div style="font-size:0.84rem;color:#564B52;line-height:1.5;">'
            f"{description}</div></div></div>",
            unsafe_allow_html=True,
        )

st.write("")
ui.explain(
    "<b>Keş (cache) nədir?</b> Streamlit hər dəfə bir düyməyə basdıqda "
    "səhifənin bütün kodunu yenidən işlədir. Əgər hər dəfə CSV oxunsa və "
    "model öyrədilsə, tətbiq dözülməz dərəcədə yavaş olardı. Keş “bu "
    "hesablamanı artıq etmişəm, nəticəsi budur” deyir - nəticədə səhifələr "
    "arasında keçid <b>bir neçə millisaniyə</b> çəkir."
)

st.divider()

# --------------------------------------------------------------------------
# 4. TƏKRARLANABİLİRLİK
# --------------------------------------------------------------------------

st.markdown("### 4. Təkrarlanabilirlik")

c1, c2, c3, c4 = st.columns(4)
with c1:
    ui.kpi("Təsadüfi toxum", str(meta.get("seed", config.SEED)),
           "bütün layihədə eyni")
with c2:
    ui.kpi("Seçilmiş k", str(meta.get("best_k", "-")),
           "yalnız validasiya ilə seçilib")
with c3:
    ui.kpi("SVD epoch", str(meta.get("svd_epochs", config.SVD_EPOCHS)),
           "tapşırığın tələbi")
with c4:
    ui.kpi("Hesablanma tarixi", str(meta.get("hesablanma_tarixi", "-"))[:10],
           "results/meta.json")

st.markdown(
    """
**Nəticələri sıfırdan təkrarlamaq üçün üç əmr kifayətdir:**

```bash
python data/generate_data.py     # 1. verilənləri yarat
python -m src.experiments        # 2. bütün eksperimentləri işlət
streamlit run app.py             # 3. tətbiqi aç
```

Toxum (seed = 42) hər üç addımda sabit olduğu üçün eyni kompüterdə eyni
paket versiyaları ilə **tam eyni rəqəmlər** alınır.
"""
)

ui.synthetic_footer()
