"""
Eksperimentlər
==============

Bu fayl layihənin bütün rəqəmlərini hesablayır və results/ qovluğuna
CSV kimi yazır. Tətbiq (Streamlit) heç nə hesablamır - yalnız bu
nəticələri oxuyub göstərir. Beləliklə heç bir rəqəm "uydurulmur".

DÖRD EKSPERİMENT:

1) EPOCH DİAQNOZU  ("20 epoch kifayətdirmi?")
   SVD müxtəlif epoch sayları ilə öyrədilir. Təlim RMSE-si hələ də sürətlə
   düşürsə, model AZ ÖYRƏDİLİB (under-training) deməkdir.
   Bu, tapşırıqdakı 20 epoch-un bizim kiçik verilənlər üçün niyə az
   olduğunu göstərir.

2) k SEÇİMİ (gizli faktor sayı)
   SVD k in {2, 5, 10, 20, 50} üçün TRAIN ilə öyrədilir, VALIDASIYA
   setində yoxlanılır. Hər k üçün 5 fərqli təsadüfi toxumla təkrar edilir
   və ORTALAMA götürülür - çünki bir tək ölçmədə SGD-nin təsadüfiliyi
   k-lar arasındakı fərqdən böyük olur.
   İki ayarla aparılır: 20 epoch (tapşırığın tələbi) və 100 epoch (diaqnozdan
   sonrakı ayar). Test seti bu mərhələdə toxunulmur.

3) YEKUN MÜQAYİSƏ
   Seçilmiş k ilə hər üç model TRAIN_FULL ilə öyrədilir və yalnız bir dəfə
   TEST setində qiymətləndirilir. Əsas cədvəl tapşırığın tələb etdiyi
   20 epoch ayarı ilədir; 100 epoch-lu SVD əlavə sətir kimi göstərilir.

4) SOYUQ START
   30 təsadüfi müştərinin təlim reytinqləri 3, 5, 10 ədədə qədər azaldılır,
   modellər yenidən öyrədilir və Precision@10 ölçülür.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from . import evaluation as ev
from .data_loader import load_all
from .models.item_cf import ItemCFRecommender
from .models.popularity import PopularityRecommender
from .models.svd import SVDRecommender
from .split import SplitResult, split_ratings


# --------------------------------------------------------------------------
# KÖMƏKÇİ
# --------------------------------------------------------------------------

def build_models(
    train: pd.DataFrame,
    all_film_ids: np.ndarray,
    n_factors: int,
    n_epochs: int = config.SVD_EPOCHS,
) -> dict:
    """Üç modeli eyni təlim verilənləri ilə öyrədir."""
    return {
        "popularity": PopularityRecommender().fit(train),
        "item_cf": ItemCFRecommender().fit(train, all_film_ids=all_film_ids),
        "svd": SVDRecommender(n_factors=n_factors, n_epochs=n_epochs).fit(
            train, all_film_ids=all_film_ids
        ),
    }


# --------------------------------------------------------------------------
# EKSPERİMENT 1: EPOCH DİAQNOZU
# --------------------------------------------------------------------------

def run_epoch_sweep(
    split: SplitResult,
    all_film_ids: np.ndarray,
    n_factors: int = config.SVD_DEFAULT_FACTORS,
    epoch_grid: list[int] = config.SVD_EPOCH_GRID,
) -> pd.DataFrame:
    """Epoch sayının təlim və validasiya RMSE-sinə təsiri.

    OXUNUŞ QAYDASI:
      - təlim RMSE-si hələ düşürsə  -> model AZ öyrədilib
      - validasiya RMSE-si qalxırsa -> model HƏDDƏN ARTIQ öyrədilib (overfitting)
    """
    rows = []
    for epochs in epoch_grid:
        model = SVDRecommender(n_factors=n_factors, n_epochs=epochs).fit(
            split.train, all_film_ids=all_film_ids
        )
        row = {
            "epochs": epochs,
            "train_rmse": model.train_rmse_history[-1],
            "val_rmse": ev.rmse(model, split.val),
        }
        rows.append(row)
        print(
            f"   epochs={epochs:>3}  təlim RMSE={row['train_rmse']:.4f}  "
            f"validasiya RMSE={row['val_rmse']:.4f}"
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# EKSPERİMENT 2: k SEÇİMİ
# --------------------------------------------------------------------------

def run_k_sweep(
    split: SplitResult,
    all_film_ids: np.ndarray,
    factors_grid: list[int] = config.SVD_FACTORS_GRID,
    epoch_settings: tuple[int, ...] = (config.SVD_EPOCHS, config.SVD_EPOCHS_TUNED),
    seed_repeats: int = config.SVD_SEED_REPEATS,
) -> pd.DataFrame:
    """Hər k üçün validasiya (seçim) və test (həssaslıq) metrikaları.

    dataset sütunu:
      "validasiya" -> TRAIN ilə öyrədilib, k SEÇİMİ bunlarla aparılır
                      (seed_repeats fərqli toxumun ORTALAMASI)
      "test"       -> TRAIN_FULL ilə öyrədilib, yalnız həssaslıq əyrisi üçün
    """
    rows = []
    for epochs in epoch_settings:
        label = "tapşırıq (20)" if epochs == config.SVD_EPOCHS else f"tənzimlənmiş ({epochs})"
        print(f"   --- {epochs} epoch ---")
        for k in factors_grid:
            # --- Validasiya: bir neçə toxumla təkrar, ortalama alınır ---
            scores = []
            for offset in range(seed_repeats):
                model = SVDRecommender(
                    n_factors=k, n_epochs=epochs, seed=config.SEED + offset
                ).fit(split.train, all_film_ids=all_film_ids)
                scores.append(
                    ev.evaluate_model(
                        model, split.train, split.val, catalog_size=len(all_film_ids)
                    )
                )
            rows.append(
                {
                    "k": k,
                    "epochs": epochs,
                    "epoch_label": label,
                    "dataset": "validasiya",
                    "rmse": float(np.mean([s["rmse"] for s in scores])),
                    "rmse_std": float(np.std([s["rmse"] for s in scores])),
                    "precision_at_k": float(
                        np.mean([s["precision_at_k"] for s in scores])
                    ),
                    "recall_at_k": float(np.mean([s["recall_at_k"] for s in scores])),
                    "coverage": float(np.mean([s["coverage"] for s in scores])),
                    "n_seeds": seed_repeats,
                }
            )
            val_rmse = rows[-1]["rmse"]
            val_std = rows[-1]["rmse_std"]

            # --- Test qolu: yalnız həssaslıq əyrisi üçün, tək toxum ---
            model_test = SVDRecommender(n_factors=k, n_epochs=epochs).fit(
                split.train_full, all_film_ids=all_film_ids
            )
            scores_test = ev.evaluate_model(
                model_test, split.train_full, split.test, catalog_size=len(all_film_ids)
            )
            rows.append(
                {
                    "k": k,
                    "epochs": epochs,
                    "epoch_label": label,
                    "dataset": "test",
                    "rmse_std": 0.0,
                    "n_seeds": 1,
                    **scores_test,
                }
            )

            print(
                f"     k={k:>2}  validasiya RMSE={val_rmse:.4f} (±{val_std:.4f})  "
                f"test RMSE={scores_test['rmse']:.4f}  "
                f"test P@10={scores_test['precision_at_k']:.4f}"
            )

    return pd.DataFrame(rows)


def choose_best_k(k_sweep: pd.DataFrame, epochs: int) -> int:
    """Verilmiş epoch ayarı üçün ən kiçik VALIDASIYA RMSE-si olan k."""
    subset = k_sweep[
        (k_sweep["dataset"] == "validasiya") & (k_sweep["epochs"] == epochs)
    ]
    return int(subset.loc[subset["rmse"].idxmin(), "k"])


# --------------------------------------------------------------------------
# EKSPERİMENT 3: YEKUN MÜQAYİSƏ
# --------------------------------------------------------------------------

def run_model_comparison(
    split: SplitResult,
    all_film_ids: np.ndarray,
    n_factors: int,
    n_factors_tuned: int,
) -> pd.DataFrame:
    """Üç modeli TEST setində müqayisə edir (yekun nəticə)."""
    models = build_models(
        split.train_full, all_film_ids, n_factors, n_epochs=config.SVD_EPOCHS
    )
    scores = {
        key: ev.evaluate_model(
            model, split.train_full, split.test, catalog_size=len(all_film_ids)
        )
        for key, model in models.items()
    }
    table = ev.results_table(scores)
    table["k_factors"] = [None, None, n_factors]
    table["epochs"] = [None, None, config.SVD_EPOCHS]
    table["Ayar"] = ["-", "-", f"tapşırıq ({config.SVD_EPOCHS} epoch)"]

    for key, values in scores.items():
        print(
            f"   {config.MODEL_SHORT_AZ[key]:<12} RMSE={values['rmse']:.4f}  "
            f"P@10={values['precision_at_k']:.4f}  "
            f"R@10={values['recall_at_k']:.4f}  "
            f"əhatə={values['coverage']:.3f}"
        )

    # --- Əlavə sətir: tənzimlənmiş SVD (diaqnozdan sonra) ---
    tuned = SVDRecommender(
        n_factors=n_factors_tuned, n_epochs=config.SVD_EPOCHS_TUNED
    ).fit(split.train_full, all_film_ids=all_film_ids)
    tuned_scores = ev.evaluate_model(
        tuned, split.train_full, split.test, catalog_size=len(all_film_ids)
    )
    tuned_row = ev.results_table({"svd": tuned_scores}).iloc[0].to_dict()
    tuned_row["model_key"] = "svd_tuned"
    tuned_row["Model"] = f"SVD - tənzimlənmiş ({config.SVD_EPOCHS_TUNED} epoch)"
    tuned_row["k_factors"] = n_factors_tuned
    tuned_row["epochs"] = config.SVD_EPOCHS_TUNED
    tuned_row["Ayar"] = f"tənzimlənmiş ({config.SVD_EPOCHS_TUNED} epoch)"

    print(
        f"   {'SVD (100ep)':<12} RMSE={tuned_scores['rmse']:.4f}  "
        f"P@10={tuned_scores['precision_at_k']:.4f}  "
        f"R@10={tuned_scores['recall_at_k']:.4f}  "
        f"əhatə={tuned_scores['coverage']:.3f}"
    )

    return pd.concat([table, pd.DataFrame([tuned_row])], ignore_index=True)


# --------------------------------------------------------------------------
# EKSPERİMENT 4: SOYUQ START
# --------------------------------------------------------------------------

def make_cold_start_train(
    train_full: pd.DataFrame, cold_customers: np.ndarray, keep: int
) -> pd.DataFrame:
    """Seçilmiş müştərilərin təlim reytinqlərini ilk `keep` ədədə qədər azaldır."""
    cold = {int(c) for c in cold_customers}
    parts = []
    for cid, group in train_full.groupby("customer_id", sort=True):
        group = group.sort_values("timestamp")
        parts.append(group.head(keep) if int(cid) in cold else group)
    return pd.concat(parts).reset_index(drop=True)


def run_cold_start(
    split: SplitResult,
    all_film_ids: np.ndarray,
    n_factors: int,
    n_customers: int = config.COLD_START_N_CUSTOMERS,
    sizes: list[int] = config.COLD_START_SIZES,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Soyuq start analizi: az reytinqlə tövsiyə keyfiyyəti necə dəyişir?"""
    rng = np.random.default_rng(config.SEED)
    everyone = np.sort(split.train_full["customer_id"].unique())
    cold_customers = rng.choice(everyone, size=n_customers, replace=False)
    cold_test = split.test[split.test["customer_id"].isin(cold_customers)]

    rows = []
    scenarios = [(keep, str(keep)) for keep in sizes] + [(None, "Tam")]

    for keep, label in scenarios:
        if keep is None:
            cold_train, order = split.train_full, 999
        else:
            cold_train = make_cold_start_train(split.train_full, cold_customers, keep)
            order = keep

        models = build_models(cold_train, all_film_ids, n_factors)
        for key, model in models.items():
            scores = ev.precision_recall_at_k(
                model, cold_train, cold_test, catalog_size=len(all_film_ids)
            )
            rows.append(
                {
                    "reytinq_sayi": label,
                    "sira": order,
                    "model_key": key,
                    "Model": config.MODEL_NAMES_AZ[key],
                    "precision_at_10": scores["precision_at_k"],
                    "recall_at_10": scores["recall_at_k"],
                    "musteri_sayi": scores["n_customers_evaluated"],
                }
            )
        best = max(rows[-3:], key=lambda r: r["precision_at_10"])
        print(
            f"   {label:>4} reytinq -> ən yaxşı: "
            f"{config.MODEL_SHORT_AZ[best['model_key']]} "
            f"P@10={best['precision_at_10']:.4f}"
        )

    return pd.DataFrame(rows), cold_customers


# --------------------------------------------------------------------------
# ƏSAS AXIN
# --------------------------------------------------------------------------

def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)

    print("Verilənlər oxunur...")
    customers, films, ratings = load_all()
    all_film_ids = films["film_id"].to_numpy()
    split = split_ratings(ratings)
    print(
        f"   train={len(split.train)}  validasiya={len(split.val)}  "
        f"test={len(split.test)}"
    )

    # ---------------- 1. epoch diaqnozu ----------------
    print("\n[1/4] Diaqnoz: 20 epoch kifayətdirmi?")
    epoch_sweep = run_epoch_sweep(split, all_film_ids)

    # ---------------- 2. k seçimi ----------------
    print("\n[2/4] SVD üçün gizli faktor sayı (k) seçilir...")
    k_sweep = run_k_sweep(split, all_film_ids)
    best_k = choose_best_k(k_sweep, config.SVD_EPOCHS)
    best_k_tuned = choose_best_k(k_sweep, config.SVD_EPOCHS_TUNED)
    print(f"   -> tapşırıq ayarı ({config.SVD_EPOCHS} epoch) üçün k = {best_k}")
    print(
        f"   -> tənzimlənmiş ayar ({config.SVD_EPOCHS_TUNED} epoch) üçün "
        f"k = {best_k_tuned}"
    )

    # ---------------- 3. yekun müqayisə ----------------
    print("\n[3/4] Üç model test setində müqayisə olunur...")
    comparison = run_model_comparison(split, all_film_ids, best_k, best_k_tuned)

    # ---------------- 4. soyuq start ----------------
    print("\n[4/4] Soyuq start analizi...")
    cold_start, cold_customers = run_cold_start(split, all_film_ids, best_k)

    # ---------------- SVD təlim əyrisi ----------------
    svd_final = SVDRecommender(n_factors=best_k).fit(
        split.train_full, all_film_ids=all_film_ids
    )
    training_curve = pd.DataFrame(
        {
            "epoch": np.arange(1, len(svd_final.train_rmse_history) + 1),
            "train_rmse": svd_final.train_rmse_history,
        }
    )

    # ---------------- Yazılır ----------------
    enc = config.CSV_ENCODING
    for name, frame in {
        "epoch_sweep": epoch_sweep,
        "k_sweep": k_sweep,
        "model_comparison": comparison,
        "cold_start": cold_start,
        "svd_training_curve": training_curve,
        "split_summary": split.summary(),
    }.items():
        frame.to_csv(config.RESULTS_DIR / f"{name}.csv", index=False, encoding=enc)

    meta = {
        "seed": config.SEED,
        "best_k": best_k,
        "best_k_tuned": best_k_tuned,
        "svd_epochs": config.SVD_EPOCHS,
        "svd_epochs_tuned": config.SVD_EPOCHS_TUNED,
        "hesablanma_tarixi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_customers": int(len(customers)),
        "n_films": int(len(films)),
        "n_ratings": int(len(ratings)),
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_test": int(len(split.test)),
        "sparsity": float(1 - len(ratings) / (len(customers) * len(films))),
        "svd_lr": config.SVD_LR,
        "svd_reg": config.SVD_REG,
        "n_neighbors": config.N_NEIGHBORS,
        "shrinkage": config.SHRINKAGE,
        "damping": config.DAMPING,
        "cold_start_customers": [int(c) for c in cold_customers],
    }
    with open(config.RESULTS_DIR / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print(f"\nBütün nəticələr saxlanıldı: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
