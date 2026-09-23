"""
Zamana görə bölgü (train / validation / test)
=============================================

NİYƏ ZAMANA GÖRƏ, TƏSADÜFİ DEYİL?
    Real həyatda biz KEÇMİŞİ bilirik və GƏLƏCƏYİ proqnoz edirik.
    Əgər bölgünü təsadüfi etsək, model müştərinin gələcək reytinqini
    görüb keçmişi proqnozlaşdıra bilər - bu, süni yüksək nəticə verər.
    Buna "məlumat sızması" (data leakage) deyilir.

BÖLGÜ QAYDASI (hər müştəri üçün ayrıca):
    1. Müştərinin reytinqləri zamana görə sıralanır.
    2. SON 20%  -> TEST      (yalnız yekun qiymətləndirmədə istifadə olunur)
    3. Qalan 80% -> təlim hissəsi
    4. Təlim hissəsinin SON 10%-i -> VALIDASIYA (yalnız k seçimi üçün)
    5. Qalanı -> TRAIN

    TRAIN            VAL      TEST
    |--------------|-----|----------|
    keçmiş  ------------------>  gələcək

İki təlim dəsti qaytarılır:
    train       - k (gizli faktor sayı) seçilərkən istifadə olunur
    train_full  - train + val; yekun test qiymətləndirməsində istifadə olunur
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class SplitResult:
    """Bölgünün nəticəsi."""

    train: pd.DataFrame  # k seçimi üçün təlim
    val: pd.DataFrame  # yalnız k seçimi üçün
    test: pd.DataFrame  # yalnız yekun qiymətləndirmə üçün
    train_full: pd.DataFrame  # train + val (yekun modellər bununla öyrədilir)

    def summary(self) -> pd.DataFrame:
        """Bölgünün qısa statistikası (tətbiqdə cədvəl kimi göstərilir)."""
        total = len(self.train) + len(self.val) + len(self.test)
        rows = [
            ("Train", len(self.train)),
            ("Validasiya", len(self.val)),
            ("Test", len(self.test)),
        ]
        return pd.DataFrame(
            {
                "Dəst": [r[0] for r in rows],
                "Reytinq sayı": [r[1] for r in rows],
                "Pay": [f"{r[1] / total:.1%}" for r in rows],
            }
        )


def split_ratings(
    ratings: pd.DataFrame,
    test_frac: float = config.TEST_FRAC,
    val_frac: float = config.VAL_FRAC,
) -> SplitResult:
    """Reytinqləri hər müştəri üçün zamana görə üç hissəyə bölür."""
    ratings = ratings.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    train_pos: list[np.ndarray] = []
    val_pos: list[np.ndarray] = []
    test_pos: list[np.ndarray] = []

    for _, group in ratings.groupby("customer_id", sort=True):
        idx = group.index.to_numpy()  # artıq zamana görə sıralanıb
        n = len(idx)

        # Son 20% -> test (ən azı 1 ədəd)
        n_test = max(1, int(np.floor(n * test_frac)))
        n_train_part = n - n_test

        # Təlim hissəsinin son 10%-i -> validasiya (ən azı 1, amma train boş qalmasın)
        n_val = max(1, int(np.floor(n_train_part * val_frac)))
        n_val = min(n_val, max(0, n_train_part - 1))

        cut_val = n_train_part - n_val
        train_pos.append(idx[:cut_val])
        val_pos.append(idx[cut_val:n_train_part])
        test_pos.append(idx[n_train_part:])

    train = ratings.loc[np.concatenate(train_pos)].reset_index(drop=True)
    val = ratings.loc[np.concatenate(val_pos)].reset_index(drop=True)
    test = ratings.loc[np.concatenate(test_pos)].reset_index(drop=True)
    train_full = (
        pd.concat([train, val])
        .sort_values(["customer_id", "timestamp"])
        .reset_index(drop=True)
    )

    return SplitResult(train=train, val=val, test=test, train_full=train_full)


def check_no_leakage(result: SplitResult) -> dict[str, bool]:
    """Bölgünün düzgünlüyünü yoxlayır (testlər bu funksiyadan istifadə edir).

    Yoxlanılan şərtlər:
        1. Heç bir reytinq eyni anda iki dəstdə olmur.
        2. Hər müştəri üçün test reytinqləri train/val-dan SONRA baş verir.
        3. Dəstlərin cəmi ilkin reytinq sayına bərabərdir.
    """
    def key(df: pd.DataFrame) -> set[tuple[int, int, str]]:
        return set(
            zip(
                df["customer_id"].astype(int),
                df["film_id"].astype(int),
                df["timestamp"].astype(str),
            )
        )

    k_train, k_val, k_test = key(result.train), key(result.val), key(result.test)

    checks = {
        "train_val_kəsişmir": len(k_train & k_val) == 0,
        "train_test_kəsişmir": len(k_train & k_test) == 0,
        "val_test_kəsişmir": len(k_val & k_test) == 0,
        "train_full_düzgündür": key(result.train_full) == (k_train | k_val),
    }

    # Zaman sırası: test hər müştəri üçün ən son olmalıdır
    order_ok = True
    train_part = result.train_full
    for cid, test_group in result.test.groupby("customer_id"):
        past = train_part.loc[train_part["customer_id"] == cid, "timestamp"]
        if len(past) and test_group["timestamp"].min() < past.max():
            order_ok = False
            break
    checks["test_zamanca_sonuncudur"] = order_ok

    return checks
