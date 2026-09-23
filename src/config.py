"""
Layihənin mərkəzi parametrləri
==============================

Bütün sabitlər bir yerdədir ki, bir dəyəri dəyişdikdə o, bütün layihədə
avtomatik dəyişsin. Müdafiədə "parametrlər haradadır?" sualının cavabı budur.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# YOLLAR
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

CUSTOMERS_CSV = DATA_DIR / "customers.csv"
FILMS_CSV = DATA_DIR / "films.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"

CSV_ENCODING = "utf-8-sig"  # Excel-də Azərbaycan hərfləri düzgün görünsün deyə

# --------------------------------------------------------------------------
# TƏKRARLANABİLİRLİK
# --------------------------------------------------------------------------
SEED = 42  # Bütün təsadüfi əməliyyatlarda eyni toxum

# --------------------------------------------------------------------------
# BÖLGÜ (split) PARAMETRLƏRİ
# --------------------------------------------------------------------------
TEST_FRAC = 0.20  # Hər müştərinin SON 20% reytinqi test üçün
VAL_FRAC = 0.10  # Təlim hissəsinin SON 10%-i validasiya üçün

# --------------------------------------------------------------------------
# MODEL PARAMETRLƏRİ
# --------------------------------------------------------------------------

# Populyarlıq modeli: sönümlü orta (damped mean) düsturundakı sabit
DAMPING = 5

# Item-based əməkdaşlıq süzgəci
N_NEIGHBORS = 30  # proqnozda neçə ən oxşar film istifadə olunur
SHRINKAGE = 10  # az ortaq reytinq olduqda oxşarlığı zəiflədən sabit

# SVD (Funk SVD) - SGD ilə təlim
#
# TAPŞIRIQDA TƏLƏB OLUNAN AYARLAR: 20 epoch, lr=0.005, reg=0.02.
# Bunlar `surprise` kitabxanasının standart dəyərləridir və MovieLens-100k
# (100 000 reytinq) kimi BÖYÜK verilənlər üçün nəzərdə tutulub.
# Bizim verilənlər ~5000 reytinqdir, yəni 20 dəfə kiçikdir - buna görə
# 20 epoch model üçün KİFAYƏT ETMİR (bax: results/epoch_sweep.csv).
# Bu səbəbdən iki ayar saxlayırıq:
#     SVD_EPOCHS       - tapşırığın tələbi (əsas nəticə cədvəli bununla)
#     SVD_EPOCHS_TUNED - diaqnozdan sonra seçilmiş ayar (əlavə analiz)
SVD_EPOCHS = 20
SVD_EPOCHS_TUNED = 100
SVD_LR = 0.005  # öyrənmə addımı
SVD_REG = 0.02  # requlyarizasiya (həddindən artıq uyğunlaşmanın qarşısını alır)
SVD_FACTORS_GRID = [2, 5, 10, 20, 50]  # validasiyada yoxlanılan k dəyərləri
SVD_DEFAULT_FACTORS = 10  # eksperimentdən əvvəl istifadə olunan baza dəyər
SVD_FOLDIN_EPOCHS = 60  # yeni istifadəçi üçün fold-in addımlarının sayı

# k seçimi zamanı neçə fərqli təsadüfi toxumla təkrar edilir.
# Təkrar lazımdır, çünki bir tək ölçmədə SGD-nin təsadüfi başlanğıcı
# k-lar arasındakı fərqdən daha böyük səs-küy yaradır.
SVD_SEED_REPEATS = 5

# "Epoch sayı kifayətdirmi?" sualını yoxlayan şəbəkə
SVD_EPOCH_GRID = [5, 10, 20, 40, 60, 100, 150, 200, 300]

# --------------------------------------------------------------------------
# QİYMƏTLƏNDİRMƏ PARAMETRLƏRİ
# --------------------------------------------------------------------------
TOP_N = 10  # Precision@10 / Recall@10
LIKE_THRESHOLD = 4  # test reytinqi >= 4 olarsa film "bəyənilmiş" sayılır
MIN_TRAIN_RATINGS = 3  # namizəd film təlim setində ən azı bu qədər reytinq almalıdır

RATING_MIN, RATING_MAX = 1.0, 5.0

# --------------------------------------------------------------------------
# SOYUQ START EKSPERİMENTİ
# --------------------------------------------------------------------------
COLD_START_N_CUSTOMERS = 30  # neçə müştəri "yeni istifadəçi" kimi götürülür
COLD_START_SIZES = [3, 5, 10]  # onlara neçə reytinq saxlanılır

# --------------------------------------------------------------------------
# DİZAYN: KİNO PALİTRASI
# --------------------------------------------------------------------------
COLOR_BURGUNDY = "#7A1F3D"  # tünd bordo  -> SVD
COLOR_GOLD = "#C9962B"  # qızılı
COLOR_TEAL = "#1F6F78"  # firuzəyi   -> Əməkdaşlıq süzgəci
COLOR_SLATE = "#6B7489"  # boz-mavi   -> Populyarlıq
COLOR_INK = "#241A22"  # tünd fon mətni
COLOR_CREAM = "#F7F3EC"  # açıq fon

# Hər modelin layihə boyu DƏYİŞMƏYƏN rəngi
MODEL_COLORS = {
    "popularity": COLOR_SLATE,
    "item_cf": COLOR_TEAL,
    "svd": COLOR_BURGUNDY,
}

# Modellərin Azərbaycan dilində adları (bütün səhifələrdə eyni istifadə olunur)
MODEL_NAMES_AZ = {
    "popularity": "Populyarlıq (baza model)",
    "item_cf": "Əməkdaşlıq süzgəci (Item-CF)",
    "svd": "SVD (matris faktorizasiyası)",
}

MODEL_SHORT_AZ = {
    "popularity": "Populyarlıq",
    "item_cf": "Item-CF",
    "svd": "SVD",
}

# Bütün səhifələrdə göstərilən xəbərdarlıq
SYNTHETIC_NOTE = "Sintetik verilənlər, real şəxslərə aid deyil."

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
