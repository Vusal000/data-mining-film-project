"""
Modellərin ümumi interfeysi
===========================

Üç model də eyni dörd metodu təqdim edir. Bu sayədə qiymətləndirmə kodu
modelin daxilini bilmədən hamısını eyni şəkildə yoxlaya bilir.

    fit(train)                 - modeli təlim verilənləri ilə öyrədir
    predict(customer, film)    - bir xana üçün bal proqnozu (1-5)
    recommend(customer, ...)   - namizəd filmlərdən ən yaxşı N ədədi
    explain(customer, film)    - tövsiyənin Azərbaycan dilində izahı
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


class BaseRecommender:
    """Bütün modellərin əsas sinfi."""

    # Tətbiqdə göstərilən ad və rəng üçün açar
    key: str = "base"

    def __init__(self) -> None:
        self.global_mean: float = config.RATING_MIN
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "BaseRecommender":
        raise NotImplementedError

    def predict(self, customer_id: int, film_id: int) -> float:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def predict_many(self, customer_ids, film_ids) -> np.ndarray:
        """Bir neçə (müştəri, film) cütü üçün proqnoz.

        Sadə versiya: bir-bir çağırır. Sürətli modellər bunu öz üsulu ilə
        əvəz edir (override).
        """
        return np.array(
            [self.predict(int(c), int(f)) for c, f in zip(customer_ids, film_ids)],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    def score_candidates(self, customer_id: int, candidates: np.ndarray) -> np.ndarray:
        """Namizəd filmlərin hər biri üçün bal qaytarır."""
        return self.predict_many(np.full(len(candidates), customer_id), candidates)

    def recommend(
        self, customer_id: int, candidates: np.ndarray, n: int = config.TOP_N
    ) -> list[tuple[int, float]]:
        """Ən yüksək ballı N filmi (film_id, bal) cütləri kimi qaytarır."""
        candidates = np.asarray(candidates)
        if len(candidates) == 0:
            return []
        scores = self.score_candidates(customer_id, candidates)
        # Bərabər ballarda sabit nəticə üçün film_id-yə görə ikinci sıralama
        order = np.lexsort((candidates, -scores))[:n]
        return [(int(candidates[i]), float(scores[i])) for i in order]

    # ------------------------------------------------------------------
    def explain(self, customer_id: int, film_id: int, titles: dict[int, str]) -> str:
        """Tövsiyənin Azərbaycan dilində qısa izahı."""
        return "İzah mövcud deyil."

    # ------------------------------------------------------------------
    @staticmethod
    def _clip(value: float) -> float:
        """Proqnozu icazə verilən 1-5 aralığına sıxır."""
        return float(np.clip(value, config.RATING_MIN, config.RATING_MAX))
