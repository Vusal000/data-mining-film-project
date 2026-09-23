"""
Modellərin test verilənlərini GÖRMƏDİYİNİ təsdiqləyən testlər
=============================================================

Burada hər modelin DAXİLİNƏ baxırıq və yoxlayırıq ki, orada test
reytinqlərindən heç bir iz yoxdur. Bu, layihənin ən vacib testidir:
əgər model testi görübsə, yüksək nəticə saxtadır.
"""

import numpy as np
import pytest

from src.models.item_cf import ItemCFRecommender
from src.models.popularity import PopularityRecommender
from src.models.svd import SVDRecommender
from src import evaluation as ev


@pytest.fixture(scope="module")
def fitted(split, films):
    """Üç model də YALNIZ train_full ilə öyrədilir."""
    all_films = films["film_id"].to_numpy()
    return {
        "popularity": PopularityRecommender().fit(split.train_full),
        "item_cf": ItemCFRecommender().fit(split.train_full, all_film_ids=all_films),
        "svd": SVDRecommender(n_factors=10).fit(
            split.train_full, all_film_ids=all_films
        ),
    }


# --------------------------------------------------------------------------
# 1. POPULYARLIQ MODELİ
# --------------------------------------------------------------------------

def test_populyarliq_sayimi_yalniz_trainden_gelir(split, fitted):
    """Film reytinq sayları train_full ilə tam üst-üstə düşməlidir."""
    expected = split.train_full.groupby("film_id").size().to_dict()
    assert fitted["popularity"].counts == {int(k): int(v) for k, v in expected.items()}


def test_populyarliq_umumi_ortasi_trainden_gelir(split, fitted):
    """Ümumi orta test reytinqlərini ehtiva etməməlidir."""
    assert fitted["popularity"].global_mean == pytest.approx(
        split.train_full["rating"].mean()
    )
    # Bütün verilənlərin ortası ilə eyni OLMAMALIDIR (yəni test qarışmayıb)
    everything = (
        split.train_full["rating"].sum() + split.test["rating"].sum()
    ) / (len(split.train_full) + len(split.test))
    assert fitted["popularity"].global_mean != pytest.approx(everything)


# --------------------------------------------------------------------------
# 2. ITEM-CF MODELİ
# --------------------------------------------------------------------------

def test_item_cf_matrisinde_test_xanalari_bosdur(split, fitted):
    """Test cütlərinin hər biri modelin matrisində BOŞ qalmalıdır."""
    model = fitted["item_cf"]
    empty = 0
    for cid, fid in zip(split.test["customer_id"], split.test["film_id"]):
        u = model.imap.customer_pos.get(int(cid))
        i = model.imap.film_pos.get(int(fid))
        if u is not None and i is not None:
            assert model.mask[u, i] == 0, "Test reytinqi modelin matrisinə düşüb!"
            empty += 1
    assert empty > 0  # yoxlamanın həqiqətən işlədiyinə əminlik


def test_item_cf_muster_ortalari_trainden_gelir(split, fitted):
    """Müştəri ortaları yalnız təlim reytinqlərindən hesablanmalıdır."""
    model = fitted["item_cf"]
    train_means = split.train_full.groupby("customer_id")["rating"].mean()
    for cid, expected in train_means.items():
        u = model.imap.customer_pos[int(cid)]
        assert model.user_means[u] == pytest.approx(expected)


def test_item_cf_oxsarliq_matrisi_simmetrikdir(fitted):
    """sim(i,j) = sim(j,i) olmalıdır və diaqonal sıfırdır."""
    sim = fitted["item_cf"].similarity
    assert np.allclose(sim, sim.T)
    assert np.allclose(np.diag(sim), 0.0)


# --------------------------------------------------------------------------
# 3. SVD MODELİ
# --------------------------------------------------------------------------

def test_svd_tarixcesinde_test_reytinqi_yoxdur(split, fitted):
    """SVD-nin saxladığı müştəri tarixçəsi test cütlərini ehtiva etməməlidir."""
    model = fitted["svd"]
    test_pairs = set(
        zip(split.test["customer_id"].astype(int), split.test["film_id"].astype(int))
    )
    stored = {
        (cid, int(fid))
        for cid, items in model.user_rated.items()
        for fid, _ in items
    }
    assert stored & test_pairs == set()


def test_svd_umumi_ortasi_trainden_gelir(split, fitted):
    assert fitted["svd"].global_mean == pytest.approx(
        split.train_full["rating"].mean()
    )


def test_svd_teleim_zamani_xeta_azalir(fitted):
    """SGD işləyirsə, təlim RMSE-si sonda başlanğıcdan kiçik olmalıdır."""
    history = fitted["svd"].train_rmse_history
    assert len(history) == 20
    assert history[-1] < history[0]


def test_svd_eyni_toxumla_eyni_neticeni_verir(split, films):
    """seed=42 ilə iki dəfə öyrədilən model eyni proqnozu verməlidir."""
    all_films = films["film_id"].to_numpy()
    a = SVDRecommender(n_factors=5).fit(split.train_full, all_film_ids=all_films)
    b = SVDRecommender(n_factors=5).fit(split.train_full, all_film_ids=all_films)
    pairs = split.test.head(50)
    pred_a = a.predict_many(pairs["customer_id"], pairs["film_id"])
    pred_b = b.predict_many(pairs["customer_id"], pairs["film_id"])
    assert np.allclose(pred_a, pred_b)


# --------------------------------------------------------------------------
# 4. TÖVSİYƏ QAYDALARI (hər üç model üçün)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["popularity", "item_cf", "svd"])
def test_tovsiyeler_artiq_baxilmis_filmleri_tekrarlamir(split, fitted, key):
    """Model müştərinin təlimdə artıq qiymətləndirdiyi filmi təklif etməməlidir."""
    model = fitted[key]
    pool = ev.eligible_films(split.train_full)
    seen = ev.seen_films_by_customer(split.train_full)

    for cid in list(seen)[:25]:
        candidates = np.array([f for f in pool if f not in seen[cid]])
        recommended = {f for f, _ in model.recommend(cid, candidates, n=10)}
        assert recommended & seen[cid] == set()


@pytest.mark.parametrize("key", ["popularity", "item_cf", "svd"])
def test_proqnozlar_1_5_araliqindadir(split, fitted, key):
    """Heç bir proqnoz 1-dən kiçik və ya 5-dən böyük ola bilməz."""
    model = fitted[key]
    sample = split.test.head(300)
    preds = model.predict_many(sample["customer_id"], sample["film_id"])
    assert preds.min() >= 1.0
    assert preds.max() <= 5.0


def test_namized_filmler_en_azi_3_reytinqlidir(split):
    """Namizəd filmlərin hamısı təlimdə >= 3 reytinq almış olmalıdır."""
    counts = split.train_full.groupby("film_id").size()
    for film_id in ev.eligible_films(split.train_full):
        assert counts[film_id] >= 3
