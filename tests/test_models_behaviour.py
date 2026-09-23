"""
Modellərin davranış testləri
============================

Burada "sızma" yox, məntiqi düzgünlük yoxlanılır: izahlar doğrudurmu,
metrikalar məntiqli aralıqdadırmı, fold-in işləyirmi.
"""

import numpy as np
import pytest

from src import config
from src import evaluation as ev
from src.data_loader import film_title_map
from src.models.item_cf import ItemCFRecommender
from src.models.popularity import PopularityRecommender
from src.models.svd import SVDRecommender


@pytest.fixture(scope="module")
def setup(split, films):
    all_films = films["film_id"].to_numpy()
    return {
        "titles": film_title_map(films),
        "popularity": PopularityRecommender().fit(split.train_full),
        "item_cf": ItemCFRecommender().fit(split.train_full, all_film_ids=all_films),
        "svd": SVDRecommender(n_factors=10).fit(
            split.train_full, all_film_ids=all_films
        ),
        "all_films": all_films,
    }


# --------------------------------------------------------------------------
# İZAHLAR
# --------------------------------------------------------------------------

def test_item_cf_izahi_yalniz_beyenilmis_filme_istinad_edir(split, setup):
    """REGRESSİYA TESTİ.

    Əvvəl belə bir səhv var idi: mənfi oxşarlıq x mənfi meyl = müsbət töhfə,
    nəticədə model "bəyəndiyiniz üçün" deyib, amma müştəri həmin filmi
    əslində BƏYƏNMƏMİŞDİ. İzahda istinad edilən film mütləq müştərinin
    öz ortasından YUXARI qiymətləndirdiyi film olmalıdır.
    """
    model, titles = setup["item_cf"], setup["titles"]
    title_to_id = {v: k for k, v in titles.items()}
    pool = ev.eligible_films(split.train_full)
    seen = ev.seen_films_by_customer(split.train_full)

    checked = 0
    for cid in list(seen)[:30]:
        candidates = np.array([f for f in pool if f not in seen[cid]])
        for film_id, _ in model.recommend(cid, candidates, n=5):
            text = model.explain(cid, film_id, titles)
            if "yüksək bal verdiyiniz üçün" not in text:
                continue
            # İzahdakı film adını çıxarırıq: «...»
            name = text.split("«")[1].split("»")[0]
            source_id = title_to_id[name]

            u = model.imap.customer_pos[cid]
            j = model.imap.film_pos[source_id]
            # 1) Müştəri həmin filmi öz ortasından yuxarı qiymətləndirməlidir
            assert model.centered[u, j] > 0, f"{name}: müştəri bunu bəyənməyib"
            # 2) Oxşarlıq müsbət olmalıdır
            i = model.imap.film_pos[film_id]
            assert model.similarity[i, j] > 0, f"{name}: oxşarlıq mənfidir"
            checked += 1

    assert checked > 0, "Heç bir izah yoxlanmadı - test mənasızdır"


def test_butun_modeller_izah_qaytarir(split, setup):
    """Hər üç model boş olmayan Azərbaycan dilində izah verməlidir."""
    pool = ev.eligible_films(split.train_full)
    seen = ev.seen_films_by_customer(split.train_full)
    cid = sorted(seen)[0]
    candidates = np.array([f for f in pool if f not in seen[cid]])

    for key in ("popularity", "item_cf", "svd"):
        for film_id, _ in setup[key].recommend(cid, candidates, n=3):
            text = setup[key].explain(cid, film_id, setup["titles"])
            assert isinstance(text, str) and len(text) > 10


# --------------------------------------------------------------------------
# METRİKALAR
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["popularity", "item_cf", "svd"])
def test_metrikalar_menali_araliqdadir(split, setup, key):
    scores = ev.evaluate_model(
        setup[key], split.train_full, split.test, catalog_size=len(setup["all_films"])
    )
    assert 0.0 < scores["rmse"] < 3.0
    assert 0.0 <= scores["precision_at_k"] <= 1.0
    assert 0.0 <= scores["recall_at_k"] <= 1.0
    assert 0.0 < scores["coverage"] <= 1.0
    assert scores["n_customers_evaluated"] > 50


def test_tovsiye_sayi_10_dan_cox_deyil(split, setup):
    pool = ev.eligible_films(split.train_full)
    seen = ev.seen_films_by_customer(split.train_full)
    cid = sorted(seen)[0]
    candidates = np.array([f for f in pool if f not in seen[cid]])
    assert len(setup["svd"].recommend(cid, candidates, n=config.TOP_N)) == config.TOP_N


def test_tovsiyeler_bala_gore_azalan_siradadir(split, setup):
    pool = ev.eligible_films(split.train_full)
    seen = ev.seen_films_by_customer(split.train_full)
    cid = sorted(seen)[0]
    candidates = np.array([f for f in pool if f not in seen[cid]])
    for key in ("popularity", "item_cf", "svd"):
        scores = [s for _, s in setup[key].recommend(cid, candidates, n=10)]
        assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# SOYUQ START (fold-in)
# --------------------------------------------------------------------------

def test_foldin_yeni_istifadeci_ucun_isleyir(setup):
    """Modeldə olmayan yeni müştəri üçün fold-in tövsiyə verə bilməlidir."""
    model, titles = setup["svd"], setup["titles"]
    rated = [(int(f), 5.0) for f in model.imap.film_ids[:3]]
    candidates = model.imap.film_ids[10:120]

    recommendations = model.recommend_foldin(rated, candidates, n=10)
    assert len(recommendations) == 10
    assert all(1.0 <= score <= 5.0 for _, score in recommendations)
    # Təklif olunanlar namizədlər arasından olmalıdır
    assert all(film_id in set(candidates.tolist()) for film_id, _ in recommendations)


def test_foldin_reytinqe_reaksiya_verir(setup):
    """Fərqli bal verən iki yeni istifadəçi fərqli vektor almalıdır."""
    model = setup["svd"]
    films_subset = [int(f) for f in model.imap.film_ids[:5]]
    high = model.fold_in([(f, 5.0) for f in films_subset])
    low = model.fold_in([(f, 1.0) for f in films_subset])
    # Səxavətli istifadəçinin meyli (b_u) daha yüksək olmalıdır
    assert high[0] > low[0]


def test_foldin_bos_siyahi_ile_cokmur(setup):
    b_new, p_new = setup["svd"].fold_in([])
    assert b_new == 0.0
    assert np.allclose(p_new, 0.0)
