"""
Səhifə 8: NƏTİCƏLƏR VƏ MƏHDUDİYYƏTLƏR
=====================================

Əsas tapıntılar, layihənin dürüst məhdudiyyətləri və müdafiədə
çox güman veriləcək 10 sualın sadə cavabı.
"""

import streamlit as st

from src import config
from src import ui

ui.page_setup(
    "📌 Nəticələr və məhdudiyyətlər",
    "Nə öyrəndik, nəyi iddia edə bilmərik və müdafiəyə hazırlıq sualları",
)

results = ui.require_results()
if results is None:
    st.stop()

comparison = results["model_comparison"]
cold_start = results["cold_start"]
epoch_sweep = results["epoch_sweep"]
meta = results["meta"]

PRECISION_COL = f"Precision@{config.TOP_N}"
RECALL_COL = f"Recall@{config.TOP_N}"
COVERAGE_COL = "Kataloq əhatəsi"

main = comparison[comparison["model_key"].isin(["popularity", "item_cf", "svd"])]
tuned = comparison[comparison["model_key"] == "svd_tuned"]

best_rmse = main.loc[main["RMSE"].idxmin()]
best_precision = main.loc[main[PRECISION_COL].idxmax()]
best_coverage = main.loc[main[COVERAGE_COL].idxmax()]
svd_spec = main[main["model_key"] == "svd"].iloc[0]
tuned_row = tuned.iloc[0] if len(tuned) else None

spec_epochs = meta.get("svd_epochs")
tuned_epochs = meta.get("svd_epochs_tuned")
best_epoch_row = epoch_sweep.loc[epoch_sweep["val_rmse"].idxmin()]

# --------------------------------------------------------------------------
# 1. ƏSAS TAPINTILAR
# --------------------------------------------------------------------------

st.markdown("### 🔑 Əsas tapıntılar")

FINDINGS = [
    (
        "Tək qalib yoxdur",
        f"<b>{config.MODEL_NAMES_AZ[best_rmse['model_key']]}</b> bal proqnozunda "
        f"ən dəqiqdir (RMSE {best_rmse['RMSE']:.4f}), amma tövsiyə siyahısında "
        f"<b>{config.MODEL_NAMES_AZ[best_precision['model_key']]}</b> qalibdir "
        f"(P@10 {best_precision[PRECISION_COL]:.4f}). “Ən yaxşı model” sualı "
        "məqsəddən asılıdır.",
        config.COLOR_BURGUNDY,
    ),
    (
        "Sadə baza model gözlənildiyindən güclüdür",
        f"Populyarlıq modeli P@10-da {main.loc[main['model_key'] == 'popularity', PRECISION_COL].iloc[0]:.4f} "
        "nəticə göstərir. Səbəb qismən <b>qiymətləndirmə üsulundadır</b>: test "
        "setinə yalnız müştərinin faktiki baxdığı filmlər düşür, insanlar isə "
        "əsasən populyar filmlərə baxır.",
        config.COLOR_SLATE,
    ),
    (
        "Standart hiperparametrlər kiçik verilənlərə uyğun gəlmir",
        f"Tapşırıqdakı {spec_epochs} epoch ayarı ilə SVD <b>az öyrədilmiş</b> "
        f"qalır. Validasiya RMSE-si ən yaxşı nöqtədə "
        f"({int(best_epoch_row['epochs'])} epoch) "
        f"{best_epoch_row['val_rmse']:.4f}-dir, {spec_epochs} epoch-da isə "
        f"{epoch_sweep[epoch_sweep['epochs'] == spec_epochs]['val_rmse'].iloc[0]:.4f}. "
        "Bu, layihənin ən dəyərli metodoloji tapıntısıdır.",
        config.COLOR_GOLD,
    ),
    (
        "Gizli faktorların sayı yalnız model öyrədilibsə əhəmiyyət kəsb edir",
        f"{spec_epochs} epoch ilə k-nın dəyişməsi RMSE-yə demək olar təsir "
        f"etmir. {tuned_epochs} epoch ilə isə k artdıqca RMSE aydın şəkildə "
        "azalır. Yəni “k vacibdirmi?” sualının cavabı təlim müddətindən asılıdır.",
        config.COLOR_TEAL,
    ),
    (
        "Kataloq əhatəsi dəqiqlikdən asılı deyil",
        f"<b>{config.MODEL_NAMES_AZ[best_coverage['model_key']]}</b> kataloqun "
        f"{best_coverage[COVERAGE_COL]:.0%}-ni tövsiyələrdə göstərir, "
        f"populyarlıq modeli isə cəmi "
        f"{main.loc[main['model_key'] == 'popularity', COVERAGE_COL].iloc[0]:.0%}. "
        "Dəqiqlik metrikaları bu fərqi ümumiyyətlə görmür.",
        "#A8452F",
    ),
    (
        "Soyuq startda sadəlik qazanır",
        "Müştəri haqqında cəmi 3 reytinq bilindikdə populyarlıq modeli hər iki "
        "fərdiləşdirilmiş modeldən yaxşı nəticə verir. Reytinq sayı artdıqca "
        "fərq bağlanır.",
        "#5A1530",
    ),
]

columns = st.columns(2)
for index, (title, body, color) in enumerate(FINDINGS):
    with columns[index % 2]:
        ui.card(f"{index + 1}. {title}", body, color=color)

st.write("")

if tuned_row is not None:
    improvement = (svd_spec["RMSE"] - tuned_row["RMSE"]) / svd_spec["RMSE"]
    st.success(
        f"**Rəqəmlərlə yekun:** SVD-ni {spec_epochs} epoch əvəzinə "
        f"{tuned_epochs} epoch öyrətmək RMSE-ni {svd_spec['RMSE']:.4f} → "
        f"{tuned_row['RMSE']:.4f} (**{improvement:.1%}** yaxşılaşma) etdi və "
        f"kataloq əhatəsini "
        f"{svd_spec[COVERAGE_COL]:.0%} → {tuned_row[COVERAGE_COL]:.0%} "
        f"qaldırdı. Model dəyişmədi - yalnız **təlim müddəti** dəyişdi.",
        icon="📈",
    )

st.divider()

# --------------------------------------------------------------------------
# 2. MƏHDUDİYYƏTLƏR
# --------------------------------------------------------------------------

st.markdown("### ⚠️ Məhdudiyyətlər — nəyi iddia EDƏ BİLMƏRİK")

st.caption(
    "Dürüst elmi işin əsas göstəricisi öz nəticələrinin sərhədlərini bilməkdir."
)

LIMITATIONS = [
    (
        "Verilənlər sintetikdir",
        "Bütün müştərilər, filmlər və reytinqlər kompüter tərəfindən "
        "yaradılıb. Heç bir real şəxs, real film və ya real izləmə davranışı "
        "yoxdur. Modellər <b>bizim özümüzün qurduğumuz</b> gizli strukturu "
        "tapır - real dünyanın strukturunu yox.",
    ),
    (
        "Verilənlər kiçikdir",
        f"{meta.get('n_customers')} müştəri, {meta.get('n_films')} film, "
        f"{meta.get('n_ratings'):,}".replace(",", " ") + " reytinq. "
        "Müqayisə üçün: MovieLens-100k 20 dəfə, sənaye sistemləri isə "
        "milyonlarla dəfə böyükdür. Bir çox metod məhz böyük verilənlərdə "
        "üstünlük qazanır.",
    ),
    (
        "Nəticələr dalğalıdır",
        "Cəmi 150 müştəri ilə metrikalar təsadüfi seçimə həssasdır. "
        "Validasiya setində cəmi "
        f"{meta.get('n_val')} reytinq var - buna görə k seçimində 5 fərqli "
        "toxumun ortalamasını götürdük. Başqa toxumla bəzi sıralamalar "
        "dəyişə bilər.",
    ),
    (
        "Reytinq real davranışı tam əks etdirmir",
        "İnsan bir filmi bəyənib heç vaxt qiymət verməyə bilər, yaxud "
        "əhval-ruhiyyəsinə görə fərqli bal verə bilər. Biz yalnız "
        "<b>qeydə alınmış balları</b> görürük - baxılma müddəti, təkrar "
        "baxış, yarımçıq qoyma kimi güclü siqnallar bu verilənlərdə yoxdur.",
    ),
    (
        "Offline qiymətləndirmənin öz meyli var",
        "Test setinə yalnız müştərinin <b>baxmağı seçdiyi</b> filmlər düşür. "
        "Model gözəl bir film təklif etsə də, müştəri ona baxmayıbsa, biz "
        "bunu səhv sayırıq. Buna “missing not at random” deyilir və bu, "
        "populyarlıq modelinə süni üstünlük verir.",
    ),
    (
        "Nəticələr real bazara ümumiləşdirilə bilməz",
        "Bu iş <b>metodların müqayisəli öyrənilməsi</b> üçündür. Buradakı "
        "rəqəmlərə əsaslanaraq “Azərbaycan bazarında SVD daha yaxşıdır” kimi "
        "bir nəticə çıxarmaq <b>düzgün olmazdı</b>.",
    ),
    (
        "Zaman amili sadələşdirilib",
        "Müştəri zövqü illər ərzində dəyişir, filmlər köhnəlir, mövsümi "
        "təsirlər olur. Modellərimiz zamanı yalnız bölgüdə istifadə edir, "
        "proqnozda isə nəzərə almır.",
    ),
    (
        "Yalnız bir bölgü sınanıb",
        "Nəticələr tək bir zamana görə bölgüyə əsaslanır. Daha etibarlı "
        "qiymətləndirmə üçün bir neçə fərqli zaman kəsiyi ilə təkrarlamaq "
        "(cross-validation) lazım idi.",
    ),
]

columns = st.columns(2)
for index, (title, body) in enumerate(LIMITATIONS):
    with columns[index % 2]:
        st.markdown(
            f'<div style="background:#FFF7F2;border-left:5px solid #A8452F;'
            f'border-radius:0 10px 10px 0;padding:0.8rem 1rem;'
            f'margin-bottom:0.6rem;">'
            f'<div style="font-weight:800;color:#8A3524;font-size:0.95rem;'
            f'margin-bottom:0.25rem;">{title}</div>'
            f'<div style="font-size:0.87rem;color:#564B52;line-height:1.55;">'
            f"{body}</div></div>",
            unsafe_allow_html=True,
        )

st.divider()

# --------------------------------------------------------------------------
# 3. MÜDAFİƏ SUALLARI
# --------------------------------------------------------------------------

st.markdown("### 🎓 Müdafiəyə hazırlıq: 10 ehtimal olunan sual")

popularity_precision = main.loc[
    main["model_key"] == "popularity", PRECISION_COL
].iloc[0]
item_cf_rmse = main.loc[main["model_key"] == "item_cf", "RMSE"].iloc[0]

QUESTIONS = [
    (
        "Niyə `surprise` kitabxanasından istifadə etmədiniz?",
        "Çünki məqsəd alqoritmi <b>başa düşmək</b> idi, sadəcə çağırmaq yox. "
        "SVD-ni numpy ilə sıfırdan yazdıq: hər sətri izah edə bilirəm - "
        "başlanğıc dəyərlər, SGD yeniləmə düsturları, requlyarizasiya və "
        "yeni istifadəçi üçün fold-in. Hazır kitabxana ilə bu izah mümkün olmazdı.",
    ),
    (
        "Bölgünü niyə təsadüfi deyil, zamana görə etdiniz?",
        "Real sistemdə biz <b>keçmişi bilib gələcəyi</b> proqnoz edirik. "
        "Təsadüfi bölgüdə model müştərinin gələcək reytinqini görüb keçmişi "
        "təxmin edə bilər - bu, süni yüksək nəticə verən <b>məlumat sızmasıdır</b>. "
        "Zamana görə bölgü bunun qarşısını alır və tests/ qovluğunda bunu "
        "yoxlayan avtomatik testlər var.",
    ),
    (
        "Test setinin sızmadığına necə əminsiniz?",
        "Sadəcə inanmırıq - <b>yoxlayırıq</b>. Testlər hər modelin daxilinə "
        "baxır: Item-CF matrisində bütün test xanalarının boş olduğunu, "
        "SVD-nin saxladığı tarixçədə test cütlərinin olmadığını və "
        "populyarlıq modelinin saylarının tam olaraq təlim setinə uyğun "
        "gəldiyini təsdiqləyir. Ümumilikdə 36 test işləyir.",
    ),
    (
        f"Niyə Precision@10 belə aşağıdır ({popularity_precision:.3f})?",
        "Çünki hər müştərinin test dövründə orta hesabla cəmi bir neçə filmi "
        "var. 10 təklifdən hamısının tuş gəlməsi <b>riyazi olaraq mümkün "
        "deyil</b> - maksimum mümkün Precision@10 çox vaxt 0.3-0.6 arasındadır. "
        "Vacib olan mütləq rəqəm yox, <b>modellər arasındakı fərqdir</b>. "
        "Müqayisə üçün təsadüfi seçim təxminən 0.02 verir.",
    ),
    (
        "Ən yaxşı RMSE ilə ən yaxşı Precision niyə fərqli modellərdədir?",
        "Çünki onlar <b>fərqli suallar</b> verir. RMSE bütün test xanalarında "
        "bal proqnozunun dəqiqliyini ölçür; Precision@10 yalnız ilk 10 filmin "
        "sırasına baxır. Model orta hesabla dəqiq ola, amma ən yuxarıda səhv "
        "filmləri göstərə bilər. Üstəlik, offline qiymətləndirmə populyar "
        "filmlərə meyillidir.",
    ),
    (
        "Sıxılma (shrinkage) düzəlişi nəyə lazımdır?",
        "İki filmi cəmi 2 nəfər qiymətləndiribsə və bal oxşardırsa, kosinus "
        "oxşarlığı 1.00 çıxa bilər - amma bu təsadüfdür. "
        f"<code>sim · n/(n+{config.SHRINKAGE})</code> düsturu oxşarlığı ortaq "
        "reytinq sayına görə zəiflədir: 2 ortaq reytinqdə 0.17 dəfəyə düşür, "
        "100 ortaq reytinqdə isə demək olar toxunulmadan qalır.",
    ),
    (
        "Sönümlü orta (damped mean) nə üçündür?",
        "Adi orta az reytinqli filmləri haqsız olaraq yuxarı qaldırır: 1 nəfərin "
        f"5 bal verdiyi film 5.0 alır. Düstur hər filmə ümumi ortada olan "
        f"{config.DAMPING} “saxta” reytinq əlavə edir, beləliklə az məlumatlı "
        "film ümumi ortaya yaxın qalır, çox reytinqli film isə öz həqiqi "
        "ortasına çatır.",
    ),
    (
        "k (gizli faktor sayı) necə seçildi?",
        f"Yalnız <b>validasiya seti</b> ilə: k ∈ {config.SVD_FACTORS_GRID} "
        "dəyərləri yoxlanıldı və ən kiçik validasiya RMSE-si seçildi "
        f"(k = {meta.get('best_k')}). Hər ölçmə 5 fərqli təsadüfi toxumla "
        "təkrarlandı, çünki bir tək ölçmədə SGD-nin təsadüfiliyi k-lar "
        "arasındakı fərqdən böyük idi. Test seti bu seçimdə istifadə olunmayıb.",
    ),
    (
        "Yeni istifadəçi üçün SVD-ni necə işlədirsiniz?",
        "<b>Fold-in</b> üsulu ilə. Film vektorları (Q) və film meylləri (b_i) "
        "dondurulur, yalnız yeni istifadəçinin b_u və p_u vektoru SGD ilə "
        "öyrədilir. Bütün modeli yenidən öyrətmək lazım gəlmir - buna görə "
        "“Soyuq start” səhifəsində cavab dərhal gəlir.",
    ),
    (
        "Bu sistemi real şirkətdə necə tətbiq edərdiniz?",
        "<b>Hibrid</b> olaraq: yeni istifadəçiyə populyar filmlər, 10-15 "
        "reytinqdən sonra fərdiləşdirilmiş modelə keçid. Əlavə olaraq "
        "kataloq əhatəsinə nəzarət edərdim (yalnız dəqiqliyə baxmaq kataloqu "
        "“öldürür”) və ən əsası - <b>onlayn A/B test</b> aparardım, çünki "
        "offline metrikalar real istifadəçi davranışını tam əvəz etmir.",
    ),
]

for index, (question, answer) in enumerate(QUESTIONS, start=1):
    with st.expander(f"**{index}. {question}**"):
        st.markdown(answer, unsafe_allow_html=True)

st.divider()

# --------------------------------------------------------------------------
# 4. GƏLƏCƏK İŞ
# --------------------------------------------------------------------------

st.markdown("### 🚀 Layihəni necə inkişaf etdirmək olar?")

c1, c2, c3 = st.columns(3)
with c1:
    ui.card(
        "Hibrid model",
        "Üç modelin proqnozunu çəkili şəkildə birləşdirmək və çəkiləri "
        "validasiya seti ilə seçmək.",
        color=config.COLOR_TEAL,
    )
with c2:
    ui.card(
        "Məzmun məlumatı",
        "Janr, il və rejissor kimi film xüsusiyyətlərini modelə əlavə etmək - "
        "bu, film tərəfindəki soyuq start problemini həll edir.",
        color=config.COLOR_GOLD,
    )
with c3:
    ui.card(
        "Daha geniş qiymətləndirmə",
        "Bir neçə zaman kəsiyi ilə təkrarlamaq, NDCG və yenilik (novelty) "
        "metrikalarını əlavə etmək.",
        color=config.COLOR_BURGUNDY,
    )

ui.synthetic_footer()
