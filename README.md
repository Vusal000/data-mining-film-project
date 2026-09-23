# 🎬 Film Tövsiyə Sistemi

**Müştəri reytinqlərinə əsaslanan tövsiyə sisteminin qurulması və üç yanaşmanın
müqayisəli qiymətləndirilməsi**

Data Mining kursu üçün müstəqil layihə.

> ⚠️ **Sintetik verilənlər, real şəxslərə aid deyil.**
> Bütün müştərilər, filmlər və reytinqlər proqram tərəfindən yaradılıb.

<details>
<summary><b>English summary</b></summary>

A movie recommender system built and evaluated three ways — a popularity
baseline, item-based collaborative filtering with adjusted cosine similarity,
and a biased SVD (Funk SVD) implemented from scratch in numpy (no `surprise`
library). Includes RMSE / Precision@10 / Recall@10 / catalogue-coverage
comparison, a latent-factor sweep, explained recommendations, and a cold-start
analysis, all presented in an 8-page interactive Streamlit dashboard.
All data is synthetic. The interface language is Azerbaijani.
</details>

---

## 📌 Tədqiqat sualı

> Müştəri reytinqlərinə əsaslanan üç fərqli tövsiyə yanaşmasından hansı daha
> yaxşı nəticə verir — və **“yaxşı” dedikdə nəyi nəzərdə tuturuq?**

Müqayisə olunan üç model:

| # | Model | Qısa izahı |
|---|-------|-----------|
| 1 | **Populyarlıq** (baza) | Ən çox reytinq alan filmlər + sönümlü orta |
| 2 | **Item-CF** | Düzəldilmiş kosinus oxşarlığı + sıxılma, 30 qonşu |
| 3 | **SVD** | Meyilli Funk SVD, SGD ilə öyrədilir (numpy, sıfırdan) |

---

## 🚀 Quraşdırma və işə salma

### Windows (PowerShell)

```powershell
# 1. Layihəni yükləyin
git clone https://github.com/<istifadəçi-adı>/data-mining-film-project.git
cd data-mining-film-project

# 2. Virtual mühit yaradın və aktivləşdirin
python -m venv .venv
.venv\Scripts\Activate.ps1

# Əgər PowerShell icazə xətası verərsə, bir dəfə bunu işlədin:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 3. Paketləri quraşdırın
pip install -r requirements.txt

# 4. Tətbiqi açın
streamlit run app.py
```

### macOS / Linux

```bash
git clone https://github.com/<istifadəçi-adı>/data-mining-film-project.git
cd data-mining-film-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Brauzer avtomatik `http://localhost:8501` ünvanında açılacaq.

### Nəticələri sıfırdan təkrarlamaq

```bash
python data/generate_data.py     # 1. sintetik verilənləri yarat
python -m src.experiments        # 2. bütün eksperimentləri işlət (~3-4 dəqiqə)
python -m pytest tests -q        # 3. testləri yoxla (53 test)
streamlit run app.py             # 4. tətbiqi aç
```

---

## 📁 Qovluq quruluşu

```
data-mining-film-project/
├── app.py                      # Giriş nöqtəsi: 8 səhifəlik menyunu qurur
├── conftest.py                 # pytest üçün ümumi verilənlər (fixture)
├── requirements.txt            # Dəqiq versiyalarla paket siyahısı
│
├── data/
│   ├── generate_data.py        # Sintetik verilənləri yaradır
│   ├── customers.csv           # 150 müştəri
│   ├── films.csv               # 200 film
│   └── ratings.csv             # 5 085 reytinq
│
├── src/
│   ├── config.py               # BÜTÜN parametrlər bir yerdə (toxum, rənglər, ayarlar)
│   ├── data_loader.py          # CSV oxuyur, id ↔ matris indeksi körpüsü qurur
│   ├── split.py                # Zamana görə train / validasiya / test bölgüsü
│   ├── evaluation.py           # RMSE, MAE, Precision@10, Recall@10, əhatə
│   ├── experiments.py          # Bütün eksperimentləri işlədir → results/
│   ├── ui.py                   # Tətbiqin rəngləri, CSS və keşlənmiş funksiyaları
│   └── models/
│       ├── base.py             # Ümumi interfeys: fit / predict / recommend / explain
│       ├── popularity.py       # Model 1: baza
│       ├── item_cf.py          # Model 2: əməkdaşlıq süzgəci
│       └── svd.py              # Model 3: Funk SVD + fold-in
│
├── pages/                      # Tətbiqin 8 səhifəsi
│   ├── 0_ev.py                 # Ev: problem, tədqiqat sualı, KPI-lar
│   ├── 1_verilenler.py         # Verilənlərin təhlili (7 diaqram)
│   ├── 2_metodlar.py           # Metodların izahı (bənzətmə + sxem + düstur)
│   ├── 3_muqayise.py           # Model müqayisəsi və nəticələr
│   ├── 4_tovsiyeler.py         # Fərdi tövsiyələr + k slayderi
│   ├── 5_soyuq_start.py        # Canlı soyuq start təcrübəsi
│   ├── 6_arxitektura.py        # Sistemin arxitekturası
│   └── 7_neticeler.py          # Nəticələr, məhdudiyyətlər, 10 müdafiə sualı
│
├── tests/                      # 53 test
│   ├── test_split.py           # Bölgü düzgündürmü?
│   ├── test_no_leakage_in_models.py   # Modellər testi görübmü? (ƏN VACİB)
│   ├── test_models_behaviour.py       # İzahlar, metrikalar, fold-in
│   └── test_app_pages.py       # 8 səhifə xətasız açılırmı?
│
├── results/                    # Hesablanmış nəticələr (CSV + JSON)
└── notebooks/
    └── analiz.ipynb            # Addım-addım izahlı Jupyter dəftəri
```

---

## 🎲 Verilənlərin yaradılma qaydaları

Verilənlər təsadüfi deyil — **gizli struktura** əsasən yaradılır ki, modellərin
öyrənəcəyi real bir qanunauyğunluq olsun.

### Gizli dəyişənlər (CSV-yə YAZILMIR, modellər onları görmür)

| Dəyişən | Paylanma | Mənası |
|---------|----------|--------|
| Filmin keyfiyyəti | `Normal(0, 0.40)` | Film ümumiyyətlə yaxşıdırmı? |
| Filmin populyarlığı | `Lognormal(0, 1.10)` | Uzun quyruqlu paylanma |
| Müştərinin meyli | `Normal(0, 0.32)` | Səxavətli / sərt qiymətləndirən |
| Janr zövqü | `Normal(0, 0.50)` + bonus | 2 sevimli janr (+0.90…+1.70), 1 sevilməyən (−0.70…−1.40) |

### Film seçimi

Müştəri `u` filmi `i` seçmə ehtimalı:

```
P(i) ∝ populyarlıq_i^0.75 × exp(0.75 × janr_uyğunluğu(u,i))
```

Yəni populyar filmlər **və** müştərinin zövqünə uyğun filmlər daha çox seçilir.

### Reytinq düsturu

```
xam_bal = 3.15 + film_keyfiyyəti + müştəri_meyli + janr_uyğunluğu + Normal(0, 0.40)
bal     = clip(round(xam_bal), 1, 5)
```

### Əlavə qaydalar

- Hər müştəridə **15–60** reytinq (`triangular(15, 32, 60)` paylanması)
- Reytinq tarixi həm qeydiyyat tarixindən, həm də filmin çıxış ilindən
  **sonra** olmalıdır — bu, zamana görə bölgünü mənalı saxlayır
- Şəhər və yaş qrupu reytinqə **təsir etmir** (qəsdən) — modellərin
  demoqrafiyaya deyil, davranışa baxmasını göstərmək üçün

### Nəticə

| Göstərici | Dəyər |
|-----------|-------|
| Müştəri / Film / Reytinq | 150 / 200 / **5 085** |
| Seyrəklik (sparsity) | **83.05 %** |
| Müştəri başına reytinq | min 16, orta 34, maks 56 |
| Orta bal | 3.59 |
| “Bəyənilmiş” (≥4) pay | 54.6 % |

---

## 🔬 Metodologiya

### Zamana görə bölgü (sızma olmadan)

Hər müştəri üçün ayrıca:

```
TRAIN (3 782)      VAL (346)   TEST (957)
|----------------|-----------|-------------|
keçmiş  ------------------------>  gələcək
```

- Son **20 %** → **test** (yalnız yekun qiymətləndirmə)
- Qalanın son **10 %** → **validasiya** (yalnız `k` seçimi)
- Qalanı → **train**

**Niyə zamana görə?** Təsadüfi bölgüdə model müştərinin gələcək reytinqini
görüb keçmişi proqnozlaşdıra bilər — bu, süni yüksək nəticə verən *məlumat
sızmasıdır*. `tests/test_no_leakage_in_models.py` hər modelin daxilinə baxıb
test reytinqlərinin orada **olmadığını** təsdiqləyir.

### Qiymətləndirmə qaydaları

- **RMSE / MAE** — bütün test xanalarında bal proqnozunun dəqiqliyi
- **Precision@10 / Recall@10** — “bəyənilmiş” = test reytinqi ≥ 4
- **Kataloq əhatəsi** — bütün tövsiyələrdə kataloqun neçə %-i görünür
- **Namizəd filmlər** — təlimdə ≥ 3 reytinqi olan və müştərinin görmədiyi
  filmlər (eyni qayda hər üç modelə tətbiq olunur)

---

## 📊 Nəticələr

### Yekun müqayisə (test seti)

| Model | RMSE ↓ | MAE ↓ | P@10 ↑ | R@10 ↑ | Əhatə ↑ |
|-------|--------|-------|--------|--------|---------|
| Populyarlıq (baza) | 0.8882 | 0.7181 | **0.0759** | **0.1998** | 13.5 % |
| **Əməkdaşlıq süzgəci (Item-CF)** | **0.7908** | **0.6222** | 0.0545 | 0.1851 | **88.0 %** |
| SVD (k=20, 20 epoch — tapşırıq ayarı) | 0.8468 | 0.6710 | 0.0421 | 0.1335 | 14.0 % |
| SVD (k=50, 100 epoch — tənzimlənmiş) | 0.7983 | 0.6266 | 0.0572 | 0.1845 | 52.5 % |

**Tək qalib yoxdur:** Item-CF bal proqnozunda, populyarlıq isə tövsiyə
siyahısında öndədir. Səbəbi tətbiqin “Model müqayisəsi” səhifəsində rəqəmlərlə
izah olunub (qısaca: offline qiymətləndirmə populyar filmlərə meyillidir və
Item-CF kataloq ortasından az tanınan filmləri önə çəkir).

### Əsas metodoloji tapıntı: 20 epoch kifayət etmir

| Epoch | Təlim RMSE | Validasiya RMSE |
|-------|-----------|-----------------|
| 5 | 0.8455 | 0.8483 |
| **20** (tapşırıq) | 0.7663 | 0.8178 |
| 60 | 0.6182 | 0.8084 |
| 100 | 0.4487 | 0.7930 |
| **150** | 0.3496 | **0.7863** ← ən yaxşı |
| 300 | 0.2557 | 0.8031 ← həddən artıq öyrənmə |

Tapşırıqda tələb olunan `20 epoch, lr=0.005, reg=0.02` dəyərləri `surprise`
kitabxanasının standart ayarlarıdır və MovieLens-100k (100 000 reytinq) kimi
verilənlər üçün nəzərdə tutulub. Bizim verilənlər ~20 dəfə kiçikdir, ona görə
model həmin nöqtədə **hələ tam öyrənməyib**. Buna görə hesabatda hər iki ayar
göstərilir: tapşırığın tələbi (əsas cədvəl) və tənzimlənmiş variant (əlavə analiz).

### Gizli faktor sayının (k) təsiri

`k ∈ {2, 5, 10, 20, 50}` yalnız **validasiya** ilə yoxlandı, hər dəyər
**5 fərqli təsadüfi toxumla** təkrarlandı (bir tək ölçmədə SGD-nin
təsadüfiliyi k-lar arasındakı fərqdən böyük olur).

- **20 epoch** ilə əyri demək olar düzdür → k-nın təsiri yoxdur (seçilən k = 20)
- **100 epoch** ilə əyri aydın aşağı enir → gizli faktorlar işləyir (seçilən k = 50)

### Soyuq start (Precision@10, 29 müştəri)

| Bilinən reytinq | Populyarlıq | Item-CF | SVD |
|-----------------|-------------|---------|-----|
| 3 | **0.0621** | 0.0034 | 0.0241 |
| 5 | **0.0621** | 0.0207 | 0.0172 |
| 10 | **0.0655** | 0.0276 | 0.0207 |
| Tam (~25) | **0.0759** | 0.0310 | 0.0276 |

Müştəri haqqında nə qədər çox bilsək, nəticə bir o qədər yaxşıdır. Ən az
məlumat olan nöqtədə **populyarlıq modeli qalibdir** — çünki o, müştərini
tanımağa ehtiyac duymur. Praktiki nəticə: yeni istifadəçiyə əvvəlcə populyar
filmlər, 10–15 reytinqdən sonra fərdiləşdirilmiş modelə keçid (**hibrid strategiya**).

---

## 🔁 Təkrarlanabilirlik

| Parametr | Dəyər |
|----------|-------|
| Təsadüfi toxum (seed) | **42** (bütün layihədə, `src/config.py`) |
| Python | 3.11+ (layihə 3.14.2 ilə hazırlanıb) |
| numpy / pandas | 2.4.2 / 3.0.1 |
| streamlit / plotly | 1.57.0 / 6.6.0 |

**İcra ardıcıllığı:** `generate_data.py` → `src.experiments` → `app.py`

Toxum hər üç addımda sabit olduğu üçün eyni paket versiyaları ilə **tam eyni
rəqəmlər** alınır. Bütün hesablanmış nəticələr `results/` qovluğunda CSV kimi
saxlanılır — **tətbiq heç bir metrikanı özü hesablamır**, yalnız bu faylları
oxuyur. Beləliklə ekranda görünən hər rəqəm nəzarətli şəraitdə ölçülmüş
həqiqi dəyərdir.

---

## ✅ Testlər

```bash
python -m pytest tests -q
```

**53 test**, o cümlədən:

- test dəstinin heç bir modelə sızmadığı (hər modelin daxilinə baxaraq)
- bölgünün zaman ardıcıllığı və təkrarlanabilirliyi
- tövsiyələrin artıq baxılmış filmləri təkrarlamaması
- izahların doğruluğu (yalnız müştərinin həqiqətən bəyəndiyi filmə istinad)
- hər 8 səhifənin xətasız açılması

---

## 📝 Qeydlər və məhdudiyyətlər

- Verilənlər **tamamilə sintetikdir** — real şəxs, film və ya davranış deyil
- Verilənlər **kiçikdir** (150 müştəri); nəticələr təsadüfi seçimə həssasdır
- Reytinq real davranışı tam əks etdirmir (baxılma müddəti, yarımçıq qoyma yoxdur)
- Offline qiymətləndirmənin öz meyli var (“missing not at random”)
- Nəticələr **real bazara ümumiləşdirilə bilməz** — bu iş metodların
  müqayisəli öyrənilməsi üçündür

Ətraflı siyahı tətbiqin **“Nəticələr və məhdudiyyətlər”** səhifəsindədir.

---

## 🛠️ İstifadə olunan texnologiyalar

Python · numpy · pandas · Streamlit · Plotly · pytest · Jupyter

`surprise` kitabxanası **qəsdən istifadə olunmayıb** — SVD alqoritmi
`src/models/svd.py` faylında numpy ilə sıfırdan yazılıb ki, hər addımı
izah etmək mümkün olsun.
