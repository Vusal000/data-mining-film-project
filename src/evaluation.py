"""
Qiymətləndirmə metrikaları
==========================

ÜÇ SUALA CAVAB VERİRİK:

1) RMSE - "Verdiyi bal proqnozu nə qədər dəqiqdir?"
       RMSE = sqrt( ortalama( (həqiqi bal - proqnoz)^2 ) )
       Kiçik olması yaxşıdır. 0.90 = orta hesabla təxminən 0.9 bal səhv.

2) Precision@10 - "Təklif etdiyim 10 filmdən neçəsi doğrudan bəyənilib?"
       Precision@10 = (bəyənilmiş tövsiyələrin sayı) / 10

3) Recall@10 - "Müştərinin bəyəndiyi filmlərin neçə faizini tuta bildim?"
       Recall@10 = (bəyənilmiş tövsiyələrin sayı) / (bəyənilə biləcək filmlərin sayı)

   "Bəyənilmiş" = test setindəki reytinq >= 4.

4) Kataloq əhatəsi (coverage) - "Kataloqun neçə faizini ümumiyyətlə təklif edirəm?"
       Aşağı əhatə = model həmişə eyni bir neçə filmi təkrarlayır.

NAMİZƏD FİLMLƏR (çox vacib qayda):
    Bir müştəriyə yalnız o filmlər təklif oluna bilər ki:
      - təlim setində ən azı 3 reytinq almış olsun (tanınmayan filmi
        heç bir model qiymətləndirə bilməz), VƏ
      - müştəri onu əvvəllər qiymətləndirməmiş olsun.
    Eyni qayda hər üç modelə tətbiq olunur - müqayisə ədalətli olsun deyə.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# 1. BAL PROQNOZUNUN DƏQİQLİYİ
# --------------------------------------------------------------------------

def rmse(model, evaluation_set: pd.DataFrame) -> float:
    """Model ilə qiymətləndirmə setinin RMSE dəyəri."""
    if len(evaluation_set) == 0:
        return float("nan")
    predictions = model.predict_many(
        evaluation_set["customer_id"].to_numpy(),
        evaluation_set["film_id"].to_numpy(),
    )
    actual = evaluation_set["rating"].to_numpy(dtype=np.float64)
    return float(np.sqrt(np.mean((actual - predictions) ** 2)))


def mae(model, evaluation_set: pd.DataFrame) -> float:
    """Orta mütləq xəta - RMSE-nin daha asan başa düşülən qardaşı."""
    if len(evaluation_set) == 0:
        return float("nan")
    predictions = model.predict_many(
        evaluation_set["customer_id"].to_numpy(),
        evaluation_set["film_id"].to_numpy(),
    )
    actual = evaluation_set["rating"].to_numpy(dtype=np.float64)
    return float(np.mean(np.abs(actual - predictions)))


# --------------------------------------------------------------------------
# 2. NAMİZƏD FİLMLƏR
# --------------------------------------------------------------------------

def eligible_films(
    train: pd.DataFrame, min_ratings: int = config.MIN_TRAIN_RATINGS
) -> np.ndarray:
    """Təlim setində ən azı `min_ratings` reytinq almış filmlər."""
    counts = train.groupby("film_id").size()
    return np.sort(counts[counts >= min_ratings].index.to_numpy())


def seen_films_by_customer(train: pd.DataFrame) -> dict[int, set[int]]:
    """Hər müştərinin təlimdə artıq qiymətləndirdiyi filmlər."""
    return {
        int(cid): set(group["film_id"].astype(int))
        for cid, group in train.groupby("customer_id")
    }


# --------------------------------------------------------------------------
# 3. TÖVSİYƏ KEYFİYYƏTİ
# --------------------------------------------------------------------------

def precision_recall_at_k(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = config.TOP_N,
    like_threshold: int = config.LIKE_THRESHOLD,
    min_ratings: int = config.MIN_TRAIN_RATINGS,
    catalog_size: int | None = None,
) -> dict[str, float]:
    """Precision@k, Recall@k və kataloq əhatəsini hesablayır.

    Qayda: hər müştəri üçün namizədlər = (təlimdə >=3 reytinqi olan filmlər)
    minus (müştərinin təlimdə artıq gördüyü filmlər).
    """
    pool = eligible_films(train, min_ratings)
    seen = seen_films_by_customer(train)

    precisions: list[float] = []
    recalls: list[float] = []
    recommended_films: set[int] = set()
    evaluated_customers = 0

    for cid, group in test.groupby("customer_id"):
        cid = int(cid)
        candidates = np.array([f for f in pool if f not in seen.get(cid, set())])
        if len(candidates) == 0:
            continue

        # Həqiqətən bəyənilmiş və təklif oluna bilən filmlər
        liked = group.loc[group["rating"] >= like_threshold, "film_id"].astype(int)
        relevant = set(liked) & set(candidates.tolist())
        if not relevant:
            # Bu müştərinin testində bəyəndiyi film yoxdursa, Recall təyin olunmur
            continue

        top = model.recommend(cid, candidates, n=k)
        top_ids = [film_id for film_id, _ in top]
        recommended_films.update(top_ids)

        hits = len(set(top_ids) & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
        evaluated_customers += 1

    if catalog_size is None:
        catalog_size = len(pool)

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else float("nan"),
        "recall_at_k": float(np.mean(recalls)) if recalls else float("nan"),
        "coverage": len(recommended_films) / catalog_size if catalog_size else float("nan"),
        "n_customers_evaluated": evaluated_customers,
    }


# --------------------------------------------------------------------------
# 4. HAMISI BİR YERDƏ
# --------------------------------------------------------------------------

def evaluate_model(
    model,
    train: pd.DataFrame,
    evaluation_set: pd.DataFrame,
    k: int = config.TOP_N,
    catalog_size: int | None = None,
) -> dict[str, float]:
    """Bir model üçün bütün metrikaları hesablayır."""
    result: dict[str, float] = {
        "rmse": rmse(model, evaluation_set),
        "mae": mae(model, evaluation_set),
    }
    result.update(
        precision_recall_at_k(
            model, train, evaluation_set, k=k, catalog_size=catalog_size
        )
    )
    return result


def results_table(scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Model nəticələrini oxunaqlı cədvələ çevirir (tətbiqdə göstərilir)."""
    rows = []
    for key, values in scores.items():
        rows.append(
            {
                "model_key": key,
                "Model": config.MODEL_NAMES_AZ.get(key, key),
                "RMSE": values.get("rmse"),
                "MAE": values.get("mae"),
                f"Precision@{config.TOP_N}": values.get("precision_at_k"),
                f"Recall@{config.TOP_N}": values.get("recall_at_k"),
                "Kataloq əhatəsi": values.get("coverage"),
            }
        )
    return pd.DataFrame(rows)
