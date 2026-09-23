"""
Model 1: Populyarlıq (baza model)
=================================

GÜNDƏLİK BƏNZƏTMƏ:
    Kinoteatrda "ən çox baxılan filmlər" lövhəsi. Heç kimi tanımır,
    hamıya eyni siyahını təklif edir.

İKİ İŞİ VAR:
    1) TÖVSİYƏ: ən çox reytinq almış filmlər (populyarlıq = reytinq sayı)
    2) BAL PROQNOZU: filmin "sönümlü ortası" (damped mean)

SÖNÜMLÜ ORTA NƏDİR VƏ NİYƏ LAZIMDIR?
    Adi orta ədalətsizdir: cəmi 1 nəfər 5 bal veribsə, filmin ortası 5.0 olur
    və o, 500 nəfərin 4.6 verdiyi filmdən yuxarı qalxır.

    Düstur:      (reytinqlərin cəmi + 5 * ümumi orta)
                 -------------------------------------
                      (reytinq sayı + 5)

    Bu, sanki hər filmə əvvəlcədən "ümumi orta"da olan 5 saxta reytinq
    əlavə edir. Az reytinqi olan film ümumi ortaya yaxın qalır,
    çox reytinqi olan film isə öz həqiqi ortasına çatır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Ən çox reytinq alan filmləri tövsiyə edən baza model."""

    key = "popularity"

    def __init__(self, damping: int = config.DAMPING) -> None:
        super().__init__()
        self.damping = damping
        self.counts: dict[int, int] = {}  # film_id -> reytinq sayı
        self.damped_means: dict[int, float] = {}  # film_id -> sönümlü orta

    # ------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        """Təlim setindən film statistikasını hesablayır."""
        self.global_mean = float(train["rating"].mean())

        stats = train.groupby("film_id")["rating"].agg(["sum", "count"])

        # Sönümlü orta: (cəm + damping * ümumi orta) / (say + damping)
        damped = (stats["sum"] + self.damping * self.global_mean) / (
            stats["count"] + self.damping
        )

        self.counts = stats["count"].astype(int).to_dict()
        self.damped_means = damped.to_dict()
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, customer_id: int, film_id: int) -> float:
        """Bal proqnozu. Diqqət: müştəri kim olursa olsun, cavab eynidir."""
        return self._clip(self.damped_means.get(int(film_id), self.global_mean))

    def predict_many(self, customer_ids, film_ids) -> np.ndarray:
        """Sürətli versiya: müştəri nəzərə alınmadığı üçün yalnız filmə baxırıq."""
        return np.array(
            [self.damped_means.get(int(f), self.global_mean) for f in film_ids],
            dtype=np.float64,
        ).clip(config.RATING_MIN, config.RATING_MAX)

    # ------------------------------------------------------------------
    def score_candidates(self, customer_id: int, candidates: np.ndarray) -> np.ndarray:
        """TÖVSİYƏ üçün bal = reytinq sayı (populyarlıq), bal proqnozu deyil.

        Tapşırıq belə tələb edir: "tövsiyələr = ən çox reytinq alan filmlər".
        """
        return np.array(
            [self.counts.get(int(f), 0) for f in candidates], dtype=np.float64
        )

    # ------------------------------------------------------------------
    def explain(self, customer_id: int, film_id: int, titles: dict[int, str]) -> str:
        n = self.counts.get(int(film_id), 0)
        return (
            f"Bu filmi {n} müştəri qiymətləndirib - kataloqun ən populyar "
            f"filmlərindəndir."
        )

    # ------------------------------------------------------------------
    def top_films(self, n: int = 10) -> list[tuple[int, int]]:
        """Ümumi ən populyar N film: (film_id, reytinq sayı)."""
        items = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(int(f), int(c)) for f, c in items[:n]]
