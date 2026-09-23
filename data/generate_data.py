"""
Sintetik verilənlərin yaradılması
=================================

Bu skript layihə üçün üç CSV faylı yaradır:
    customers.csv  - 150 müştəri
    films.csv      - 200 film
    ratings.csv    - ~5000 reytinq (1-5 arası tam ədəd)

DİQQƏT: Bütün verilənlər süni (sintetik) şəkildə yaradılır.
"Sintetik verilənlər, real şəxslərə aid deyil."

Yaradılma məntiqi (README.md-də də izah olunur):
-----------------------------------------------
1. Hər filmin GİZLİ KEYFİYYƏTİ var          -> quality ~ Normal(0, 0.50)
2. Hər filmin POPULYARLIĞI var              -> uzun quyruqlu (lognormal) paylanma
3. Hər müştərinin JANR ZÖVQÜ var            -> 10 janr üzrə meyl vektoru
4. Hər müştərinin ŞƏXSİ MEYLİ (bias) var    -> kimisi səxavətli, kimisi sərt qiymət verir
5. Müştəri film seçərkən: populyar filmlər + öz zövqünə uyğun janrlar daha çox şans alır
6. Reytinq = baza (3.30) + film keyfiyyəti + müştəri meyli + janr uyğunluğu + səs-küy
   Nəticə 1-5 aralığına yuvarlaqlaşdırılır.

Bu gizli struktura görə modellər real olaraq nəsə "öyrənə" bilir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# SABİTLƏR
# --------------------------------------------------------------------------

SEED = 42  # Təkrarlanabilirlik üçün bütün layihədə eyni toxum istifadə olunur

N_CUSTOMERS = 150
N_FILMS = 200

# Baza qiymət. Müştərilər öz zövqünə uyğun film seçdiyi üçün son orta reytinq
# bu rəqəmdən yuxarı çıxır - buna görə baza 3.5 deyil, aşağı götürülüb.
GLOBAL_MEAN = 3.15

# Reytinq düsturunun komponentlərinin gücü
#
# NİYƏ MƏHZ BU NİSBƏTLƏR?
#   Əgər film keyfiyyəti və müştəri meyli (yəni "bias" hədləri) çox güclü olsa,
#   reytinqlərin demək olar hamısını bu iki sadə ədəd izah edir və GİZLİ
#   FAKTORLARA (janr zövqünə) iş qalmır. Onda SVD-də k-nın artırılması nəticəyə
#   təsir etmir. Biz janr zövqünü qəsdən güclü, bias hədlərini isə orta
#   səviyyədə saxlayırıq ki, matris faktorizasiyasının öyrənəcəyi real
#   struktur olsun.
QUALITY_SD = 0.40  # film keyfiyyətinin yayılması
BIAS_SD = 0.32  # müştəri şəxsi meylinin yayılması
NOISE_SD = 0.40  # təsadüfi səs-küy (nə qədər kiçikdirsə, struktur o qədər öyrənilə bilir)
GENRE_PREF_SD = 0.50  # janr zövqünün baza yayılması (GÜCLƏNDİRİLİB)
POPULARITY_SIGMA = 1.10  # nə qədər böyükdürsə, "uzun quyruq" bir o qədər kəskindir

# Film seçimində janr zövqünün çəkisi.
# Qəsdən reytinq düsturundakı təsirdən ZƏİFDİR: müştəri yalnız öz sevimli
# janrına baxsaydı, verilənlərdə janrlar arası fərq görünməz olardı.
TASTE_WEIGHT = 0.75
# Film seçimində populyarlığın çəkisi (1.0 = tam populyarlıq, 0.0 = populyarlıq təsirsiz)
POPULARITY_WEIGHT = 0.80

# Hər müştəriyə neçə reytinq düşür: triangular(15, 32, 60) -> orta ~35.7
# Tapşırığın tələbi: hər müştəridə 15-60 reytinq, ümumilikdə 4500-5500.
MIN_RATINGS, MODE_RATINGS, MAX_RATINGS = 15, 32, 60

SIGNUP_START = pd.Timestamp("2021-01-01")
SIGNUP_END = pd.Timestamp("2025-06-30")
RATING_END = pd.Timestamp("2026-06-30")  # reytinqlər bu tarixdən sonra ola bilməz

CITIES = ["Bakı", "Gəncə", "Sumqayıt", "Mingəçevir", "Şəki", "Lənkəran"]
# Bakı ən böyük şəhər olduğu üçün müştərilərin çoxu oradandır
CITY_WEIGHTS = [0.45, 0.16, 0.15, 0.09, 0.08, 0.07]

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_WEIGHTS = [0.22, 0.31, 0.24, 0.15, 0.08]

GENRES = [
    "Döyüş",
    "Komediya",
    "Dram",
    "Qorxu",
    "Romantika",
    "Elmi-fantastika",
    "Triller",
    "Animasiya",
    "Sənədli",
    "Macəra",
]
# Janrların kataloqda görünmə tezliyi (Dram və Komediya daha çox yayılıb)
GENRE_WEIGHTS = [0.12, 0.15, 0.18, 0.07, 0.11, 0.08, 0.10, 0.06, 0.05, 0.08]

# --------------------------------------------------------------------------
# AD SİYAHILARI (uydurma, real şəxslərə aid deyil)
# --------------------------------------------------------------------------

MALE_NAMES = [
    "Rəşad", "Elvin", "Kamran", "Orxan", "Tural", "Murad", "Elçin", "Zaur",
    "Vüqar", "Ramin", "Sənan", "Cavid", "Fərid", "Nurlan", "Emin", "Anar",
    "Rüfət", "Samir", "Toğrul", "İlkin", "Əli", "Kənan", "Cəlal", "Rövşən",
    "Səbuhi", "Fuad", "Nicat", "Ayxan", "Rəvan", "Elmar",
]

FEMALE_NAMES = [
    "Aysel", "Nigar", "Leyla", "Günel", "Nərmin", "Səbinə", "Fidan", "Ülviyyə",
    "Aytən", "Lalə", "Aynur", "Gülnar", "Türkan", "Xanım", "Şəbnəm", "Mələk",
    "Nuranə", "Zeynəb", "Ayan", "Sevinc", "Könül", "Ruhiyyə", "Aygün", "Xəyalə",
    "Pərvanə", "Səidə", "Ulduz", "Rəna", "Mehriban", "Vüsalə",
]

# Soyad kökləri: kişi üçün "-ov/-yev", qadın üçün "-ova/-yeva" şəkilçisi əlavə olunur
SURNAME_STEMS = [
    ("Məmməd", "ov"), ("Əli", "yev"), ("Hüseyn", "ov"), ("Quli", "yev"),
    ("Həsən", "ov"), ("İsmayıl", "ov"), ("Rəhim", "ov"), ("Kərim", "ov"),
    ("Abbas", "ov"), ("Nəbi", "yev"), ("Süleyman", "ov"), ("Baba", "yev"),
    ("Şirin", "ov"), ("Vəli", "yev"), ("Oruc", "ov"), ("Musa", "yev"),
    ("Cəfər", "ov"), ("Xəlil", "ov"), ("Sadıq", "ov"), ("Nəsir", "ov"),
    ("Salman", "ov"), ("Tağı", "yev"), ("Zeynal", "ov"), ("Əhməd", "ov"),
    ("Mahmud", "ov"), ("Rza", "yev"), ("Şükür", "ov"), ("Novruz", "ov"),
]

# --------------------------------------------------------------------------
# FİLM ADI ÜÇÜN SÖZ SİYAHILARI (uydurma adlar)
# --------------------------------------------------------------------------

TITLE_ADJECTIVES = [
    "Səssiz", "Qırmızı", "Qara", "Ağ", "Mavi", "Sonuncu", "Uzaq", "Gizli",
    "Soyuq", "Köhnə", "Yeni", "Sınıq", "Unudulmuş", "İtirilmiş", "Dərin",
    "Qarlı", "Yağışlı", "Küləkli", "Şirin", "Acı", "Boş", "Uzun", "Parlaq",
    "Qaranlıq", "Sakit", "Dəli", "Cəsur", "Azad", "Yalnız", "Əbədi",
    "Ölümsüz", "Sehrli", "Gümüşü", "Qızılı", "Kölgəli", "İsti", "Yüngül",
]

TITLE_NOUNS = [
    "Liman", "Qatar", "Küçə", "Şəhər", "Kənd", "Bağ", "Ev", "Qapı",
    "Pəncərə", "Məktub", "Kölgə", "İşıq", "Yol", "Körpü", "Dəniz", "Dağ",
    "Meşə", "Çay", "Göl", "Qala", "Saray", "Otaq", "Səs", "Nəfəs",
    "Yuxu", "Xatirə", "Sirr", "Nağıl", "Mahnı", "Rəqs", "Zəng", "Saat",
    "Güzgü", "Açar", "Pilləkən", "Sahil", "Səhra", "Tufan", "Bahar",
    "Payız", "Ulduz", "Külək", "Qum", "Buz", "Alov", "Bulud", "Damla",
]


# --------------------------------------------------------------------------
# 1. MÜŞTƏRİLƏR
# --------------------------------------------------------------------------

def build_customers(rng: np.random.Generator) -> pd.DataFrame:
    """150 müştəri yaradır: ad, yaş qrupu, şəhər, qeydiyyat tarixi."""
    names: list[str] = []
    seen: set[str] = set()

    # Təkrarlanmayan ad-soyad kombinasiyaları yığırıq
    while len(names) < N_CUSTOMERS:
        is_female = rng.random() < 0.5
        first = rng.choice(FEMALE_NAMES if is_female else MALE_NAMES)
        stem, suffix = SURNAME_STEMS[rng.integers(len(SURNAME_STEMS))]
        # Qadın soyadına "a" hərfi əlavə olunur: Məmmədov -> Məmmədova
        last = stem + suffix + ("a" if is_female else "")
        full = f"{first} {last}"
        if full not in seen:
            seen.add(full)
            names.append(full)

    # Qeydiyyat tarixi: 2021-01-01 ilə 2025-06-30 arasında təsadüfi gün
    span_days = (SIGNUP_END - SIGNUP_START).days
    offsets = rng.integers(0, span_days + 1, size=N_CUSTOMERS)
    signup_dates = [SIGNUP_START + pd.Timedelta(days=int(d)) for d in offsets]

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, N_CUSTOMERS + 1),
            "name": names,
            "age_group": rng.choice(AGE_GROUPS, size=N_CUSTOMERS, p=AGE_WEIGHTS),
            "city": rng.choice(CITIES, size=N_CUSTOMERS, p=CITY_WEIGHTS),
            "signup_date": [d.strftime("%Y-%m-%d") for d in signup_dates],
        }
    )


# --------------------------------------------------------------------------
# 2. FİLMLƏR
# --------------------------------------------------------------------------

def build_films(rng: np.random.Generator) -> pd.DataFrame:
    """200 film yaradır: ad, il, 1-2 janr.

    Əlavə olaraq iki GİZLİ sütun qaytarılır:
        _quality    - filmin gizli keyfiyyəti (reytinq düsturunda istifadə olunur)
        _popularity - filmin populyarlığı (seçilmə ehtimalını müəyyən edir)
    Bu sütunlar CSV-yə yazılmır, çünki modellər onları bilməməlidir.
    """
    titles: list[str] = []
    seen: set[str] = set()

    # Təkrarlanmayan uydurma adlar: "Sifət + İsim" (məs. "Səssiz Liman")
    while len(titles) < N_FILMS:
        adj = TITLE_ADJECTIVES[rng.integers(len(TITLE_ADJECTIVES))]
        noun = TITLE_NOUNS[rng.integers(len(TITLE_NOUNS))]
        title = f"{adj} {noun}"
        if title not in seen:
            seen.add(title)
            titles.append(title)

    # İstehsal ili: son illərə doğru bir az daha sıx (kataloqlar belə olur)
    years = 1990 + (rng.beta(2.2, 1.6, size=N_FILMS) * 35).astype(int)
    years = np.clip(years, 1990, 2025)

    # Janrlar: hər filmin 1 və ya 2 janrı olur
    genres_col: list[str] = []
    genre_matrix = np.zeros((N_FILMS, len(GENRES)))  # sətir: film, sütun: janr
    for i in range(N_FILMS):
        k = 1 if rng.random() < 0.45 else 2
        idx = rng.choice(len(GENRES), size=k, replace=False, p=GENRE_WEIGHTS)
        genres_col.append("|".join(GENRES[j] for j in idx))
        # Sətir cəmi 1 olsun deyə bölürük -> janr uyğunluğu ortalama kimi hesablanır
        genre_matrix[i, idx] = 1.0 / k

    films = pd.DataFrame(
        {
            "film_id": np.arange(1, N_FILMS + 1),
            "title": titles,
            "year": years,
            "genres": genres_col,
        }
    )

    # GİZLİ dəyişənlər
    films["_quality"] = rng.normal(0.0, QUALITY_SD, size=N_FILMS)
    # Uzun quyruq: bir neçə film çox populyar, çoxu az populyar
    pop = rng.lognormal(mean=0.0, sigma=POPULARITY_SIGMA, size=N_FILMS)
    films["_popularity"] = pop / pop.sum()

    return films, genre_matrix


# --------------------------------------------------------------------------
# 3. REYTİNQLƏR
# --------------------------------------------------------------------------

def build_ratings(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    films: pd.DataFrame,
    genre_matrix: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Hər müştəri üçün 15-60 reytinq yaradır.

    Qaytarır: (ratings_df, customer_bias, genre_preferences)
    """
    n_genres = len(GENRES)

    # --- Müştərinin gizli xüsusiyyətləri ---
    # Baza janr zövqü: sıfır ətrafında kiçik fərqlər
    genre_pref = rng.normal(0.0, GENRE_PREF_SD, size=(N_CUSTOMERS, n_genres))
    # Hər müştəriyə 2 "sevimli" və 1 "sevmədiyi" janr seçirik.
    # Həm müsbət, həm mənfi meyl olması gizli faktorların öyrənilməsini
    # asanlaşdırır: model yalnız "kim nəyi sevir" deyil, "kim nədən qaçır"
    # məlumatını da görür.
    for u in range(N_CUSTOMERS):
        picked = rng.choice(n_genres, size=3, replace=False)
        genre_pref[u, picked[:2]] += rng.uniform(0.90, 1.70, size=2)
        genre_pref[u, picked[2]] -= rng.uniform(0.70, 1.40)

    # Şəxsi meyl: kimisi hər filmə yüksək, kimisi aşağı bal verir
    customer_bias = rng.normal(0.0, BIAS_SD, size=N_CUSTOMERS)

    # Janr uyğunluğu matrisi: (müştəri x film)
    # genre_pref @ genre_matrix.T  ->  filmin janrları üzrə zövqün ortalaması
    affinity = genre_pref @ genre_matrix.T

    quality = films["_quality"].to_numpy()
    popularity = films["_popularity"].to_numpy()
    film_years = films["year"].to_numpy()

    # Hər müştərinin neçə film qiymətləndirəcəyi
    n_ratings = rng.triangular(
        MIN_RATINGS, MODE_RATINGS, MAX_RATINGS, size=N_CUSTOMERS
    ).astype(int)

    signup = pd.to_datetime(customers["signup_date"]).to_numpy()

    rows_customer, rows_film, rows_rating, rows_time = [], [], [], []

    for u in range(N_CUSTOMERS):
        # --- Film seçimi ---
        # Ehtimal = populyarlıq^0.8 * exp(1.2 * janr uyğunluğu)
        # Yəni: populyar filmlər daha çox seçilir, amma öz zövqünə uyğun olanlar da.
        weights = (popularity ** POPULARITY_WEIGHT) * np.exp(
            TASTE_WEIGHT * affinity[u]
        )
        weights = weights / weights.sum()
        chosen = rng.choice(N_FILMS, size=n_ratings[u], replace=False, p=weights)

        # --- Reytinq dəyəri ---
        raw = (
            GLOBAL_MEAN
            + quality[chosen]
            + customer_bias[u]
            + affinity[u, chosen]
            + rng.normal(0.0, NOISE_SD, size=len(chosen))
        )
        stars = np.clip(np.rint(raw), 1, 5).astype(int)

        # --- Zaman damğası ---
        # Reytinq tarixi: qeydiyyatdan VƏ filmin çıxış ilindən sonra olmalıdır
        film_release = pd.to_datetime(
            [f"{y}-01-01" for y in film_years[chosen]]
        ).to_numpy()
        start = np.maximum(film_release, signup[u])
        span = (RATING_END.to_numpy() - start) / np.timedelta64(1, "D")
        span = np.maximum(span, 1.0)
        offsets = rng.random(len(chosen)) * span
        times = start + (offsets * 86400).astype("timedelta64[s]")

        rows_customer.append(np.full(len(chosen), u + 1))
        rows_film.append(chosen + 1)  # film_id 1-dən başlayır
        rows_rating.append(stars)
        rows_time.append(times)

    ratings = pd.DataFrame(
        {
            "customer_id": np.concatenate(rows_customer),
            "film_id": np.concatenate(rows_film),
            "rating": np.concatenate(rows_rating),
            "timestamp": np.concatenate(rows_time),
        }
    )
    # Oxunaqlı tarix formatı
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    ratings = ratings.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    return ratings, customer_bias, genre_pref


# --------------------------------------------------------------------------
# ƏSAS FUNKSİYA
# --------------------------------------------------------------------------

def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(SEED)

    customers = build_customers(rng)
    films, genre_matrix = build_films(rng)
    ratings, _, _ = build_ratings(rng, customers, films, genre_matrix)

    # Gizli sütunları CSV-yə yazmırıq - modellər onları görməməlidir
    films_public = films.drop(columns=["_quality", "_popularity"])

    # utf-8-sig: Excel-də Azərbaycan hərfləri düzgün görünsün deyə
    customers.to_csv(out_dir / "customers.csv", index=False, encoding="utf-8-sig")
    films_public.to_csv(out_dir / "films.csv", index=False, encoding="utf-8-sig")
    ratings.to_csv(out_dir / "ratings.csv", index=False, encoding="utf-8-sig")

    # ---------------- Qısa hesabat ----------------
    per_user = ratings.groupby("customer_id").size()
    per_film = ratings.groupby("film_id").size()
    sparsity = 1 - len(ratings) / (N_CUSTOMERS * N_FILMS)

    print("=" * 58)
    print("SİNTETİK VERİLƏNLƏR HAZIRDIR")
    print("Sintetik verilənlər, real şəxslərə aid deyil.")
    print("=" * 58)
    print(f"Müştəri sayı           : {len(customers)}")
    print(f"Film sayı              : {len(films_public)}")
    print(f"Reytinq sayı           : {len(ratings)}")
    print(f"Seyrəklik (sparsity)   : {sparsity:.2%}")
    print()
    print(f"Müştəri başına reytinq : min {per_user.min()}, "
          f"orta {per_user.mean():.1f}, maks {per_user.max()}")
    print(f"Film başına reytinq    : min {per_film.min()}, "
          f"orta {per_film.mean():.1f}, maks {per_film.max()}")
    print(f"Heç reytinq almayan film: {N_FILMS - per_film.size}")
    print()
    print(f"Orta reytinq           : {ratings['rating'].mean():.3f}")
    print("Reytinq paylanması     :")
    dist = ratings["rating"].value_counts().sort_index()
    for star, cnt in dist.items():
        print(f"   {star} ulduz : {cnt:5d}  ({cnt / len(ratings):6.2%})")
    liked = (ratings["rating"] >= 4).mean()
    print(f"'Bəyənilmiş' (>=4) pay : {liked:.2%}")
    print()
    print("Ən çox reytinq alan 5 film:")
    top = per_film.sort_values(ascending=False).head(5)
    for fid, cnt in top.items():
        row = films_public.loc[films_public["film_id"] == fid].iloc[0]
        print(f"   {row['title']:<22} ({row['year']}) - {cnt} reytinq")
    print("=" * 58)


if __name__ == "__main__":
    main()
