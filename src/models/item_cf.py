"""
Model 2: Item-based əməkdaşlıq süzgəci (Item-based Collaborative Filtering)
===========================================================================

GÜNDƏLİK BƏNZƏTMƏ:
    "Bu filmi bəyənənlər həm də bunu bəyənib." Model filmlər arasında
    oxşarlıq tapır - amma janra və ya süjetə yox, İNSANLARIN
    QİYMƏTLƏNDİRMƏ DAVRANIŞINA baxaraq.

ÜÇ ADDIM:

1) DÜZƏLDİLMİŞ KOSİNUS OXŞARLIĞI (adjusted cosine)
   Problem: kimisi hər filmə 5 verir, kimisi maksimum 3 verir.
   Həll: hər müştərinin ballarından ÖZ ORTASINI çıxırıq.
         "Bu müştəri bu filmi öz adi səviyyəsindən yuxarı qiymətləndirib?"

   Düstur (i və j filmləri üçün, hər ikisini qiymətləndirən müştərilər üzrə):

        sim(i,j) = SUM_u (r_ui - ort_u)(r_uj - ort_u)
                   ----------------------------------
                   sqrt(SUM (r_ui-ort_u)^2) * sqrt(SUM (r_uj-ort_u)^2)

2) SIXILMA DÜZƏLİŞİ (shrinkage)
   Problem: 2 nəfər hər iki filmi bəyənibsə, oxşarlıq 1.0 çıxa bilər -
   amma bu təsadüf ola bilər.
   Həll:  sim_düzəldilmiş = sim * n / (n + 10)
          n = hər iki filmi qiymətləndirən müştərilərin sayı.
   Az ortaq reytinq -> oxşarlıq sıfıra doğru "sıxılır".

3) PROQNOZ (30 ən oxşar qonşu ilə)

        proqnoz(u,i) = ort_u + SUM_j sim(i,j) * (r_uj - ort_u)
                                --------------------------------
                                     SUM_j |sim(i,j)|

   j - müştərinin qiymətləndirdiyi, i-yə ən oxşar 30 film.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..data_loader import IndexMap, build_index, build_matrix
from .base import BaseRecommender


class ItemCFRecommender(BaseRecommender):
    """Film-film oxşarlığına əsaslanan tövsiyə modeli."""

    key = "item_cf"

    def __init__(
        self,
        n_neighbors: int = config.N_NEIGHBORS,
        shrinkage: int = config.SHRINKAGE,
    ) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage

        self.imap: IndexMap | None = None
        self.similarity: np.ndarray | None = None  # (film x film)
        self.centered: np.ndarray | None = None  # (müştəri x film), boşlar 0
        self.mask: np.ndarray | None = None  # hansı xanada reytinq var
        self.user_means: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, train: pd.DataFrame,
            all_film_ids: np.ndarray | None = None) -> "ItemCFRecommender":
        self.global_mean = float(train["rating"].mean())
        self.imap = build_index(train, all_film_ids=all_film_ids)

        ratings_matrix = build_matrix(train, self.imap)
        self.mask = (ratings_matrix > 0).astype(np.float64)

        # --- Addım 1a: hər müştərinin öz orta balı ---
        counts = self.mask.sum(axis=1)
        sums = ratings_matrix.sum(axis=1)
        # Reytinqi olmayan müştəri üçün ümumi ortadan istifadə edirik
        self.user_means = np.where(counts > 0, sums / np.maximum(counts, 1),
                                   self.global_mean)

        # --- Addım 1b: mərkəzləşdirmə (hər baldan öz ortasını çıxırıq) ---
        self.centered = (ratings_matrix - self.user_means[:, None]) * self.mask

        # --- Addım 1c: kosinus oxşarlığı ---
        # centered.T @ centered -> surət (yalnız ortaq qiymətləndirənlər sayılır,
        # çünki reytinqi olmayan xanalar 0-dır)
        numerator = self.centered.T @ self.centered
        norms = np.sqrt(np.diag(numerator))
        denominator = np.outer(norms, norms)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denominator > 0, numerator / denominator, 0.0)

        # --- Addım 2: sıxılma düzəlişi ---
        co_counts = self.mask.T @ self.mask  # neçə müştəri hər iki filmi qiymətləndirib
        sim = sim * (co_counts / (co_counts + self.shrinkage))

        np.fill_diagonal(sim, 0.0)  # film özü ilə müqayisə olunmur
        self.similarity = np.nan_to_num(sim)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def _predict_positions(self, u_pos: int, film_positions: np.ndarray) -> np.ndarray:
        """Bir müştəri üçün bir neçə filmin balını hesablayır (matris indeksləri ilə)."""
        assert self.similarity is not None and self.centered is not None
        assert self.mask is not None and self.user_means is not None

        user_mean = self.user_means[u_pos]
        rated = np.flatnonzero(self.mask[u_pos] > 0)  # müştərinin baxdığı filmlər

        if len(rated) == 0:
            return np.full(len(film_positions), self.global_mean)

        # (namizəd film x müştərinin baxdığı film) oxşarlıq bloku
        sims = self.similarity[np.ix_(film_positions, rated)]
        deviations = self.centered[u_pos, rated]  # r_uj - ort_u

        # Yalnız müsbət oxşarlıqlar, yalnız 30 ən yaxın qonşu - bax:
        # _weighted_prediction (eyni hesablama yeni istifadəçi üçün də işləyir)
        return self._weighted_prediction(sims, deviations, user_mean)

    # ------------------------------------------------------------------
    def predict(self, customer_id: int, film_id: int) -> float:
        assert self.imap is not None
        u_pos = self.imap.customer_pos.get(int(customer_id))
        i_pos = self.imap.film_pos.get(int(film_id))
        if u_pos is None or i_pos is None:
            # Tanınmayan müştəri və ya film -> ümumi ortaya qayıdırıq
            return self._clip(self.global_mean)
        return float(self._predict_positions(u_pos, np.array([i_pos]))[0])

    def predict_many(self, customer_ids, film_ids) -> np.ndarray:
        """Müştəri-müştəri qruplaşdıraraq sürətli hesablama."""
        assert self.imap is not None
        customer_ids = np.asarray(customer_ids, dtype=int)
        film_ids = np.asarray(film_ids, dtype=int)
        out = np.full(len(customer_ids), self.global_mean, dtype=np.float64)

        for cid in np.unique(customer_ids):
            u_pos = self.imap.customer_pos.get(int(cid))
            where = np.flatnonzero(customer_ids == cid)
            if u_pos is None:
                continue
            positions = np.array(
                [self.imap.film_pos.get(int(f), -1) for f in film_ids[where]]
            )
            known = positions >= 0
            if known.any():
                out[where[known]] = self._predict_positions(
                    u_pos, positions[known]
                )
        return out

    # ------------------------------------------------------------------
    def score_candidates(self, customer_id: int, candidates: np.ndarray) -> np.ndarray:
        assert self.imap is not None
        u_pos = self.imap.customer_pos.get(int(customer_id))
        positions = np.array([self.imap.film_pos.get(int(f), -1) for f in candidates])
        scores = np.full(len(candidates), self.global_mean, dtype=np.float64)
        known = positions >= 0
        if u_pos is not None and known.any():
            scores[known] = self._predict_positions(u_pos, positions[known])
        return scores

    # ------------------------------------------------------------------
    def explain(self, customer_id: int, film_id: int, titles: dict[int, str]) -> str:
        """Tövsiyəyə ən çox töhfə verən filmi tapıb izah qaytarır."""
        assert self.imap is not None and self.similarity is not None
        u_pos = self.imap.customer_pos.get(int(customer_id))
        i_pos = self.imap.film_pos.get(int(film_id))
        if u_pos is None or i_pos is None:
            return "Bu müştəri üçün kifayət qədər məlumat yoxdur."

        rated = np.flatnonzero(self.mask[u_pos] > 0)
        if len(rated) == 0:
            return "Bu müştəri üçün kifayət qədər məlumat yoxdur."

        sims = self.similarity[i_pos, rated]
        deviations = self.centered[u_pos, rated]

        # DİQQƏT: izah yalnız o zaman doğrudur ki, HƏM oxşarlıq müsbət olsun,
        # HƏM də müştəri həmin filmi öz ortasından YUXARI qiymətləndirmiş olsun.
        # Əks halda "iki mənfinin hasili müsbətdir" effekti yanlış izah yaradır.
        valid = (sims > 0) & (deviations > 0)
        if not valid.any():
            return "Oxşar filmlər arasında güclü uyğunluq tapılmadı."

        contribution = np.where(valid, sims * deviations, -np.inf)
        best = int(np.argmax(contribution))

        source_film_id = int(self.imap.film_ids[rated[best]])
        source_title = titles.get(source_film_id, f"#{source_film_id}")
        return (
            f"«{source_title}» filminə yüksək bal verdiyiniz üçün "
            f"(oxşarlıq {sims[best]:.2f})."
        )

    # ------------------------------------------------------------------
    # YENİ İSTİFADƏÇİ (modeldə olmayan müştəri)
    # ------------------------------------------------------------------
    def _weighted_prediction(
        self, sims: np.ndarray, deviations: np.ndarray, fallback: float
    ) -> np.ndarray:
        """Qonşuların çəkili ortası ilə proqnoz (ümumi hesablama hissəsi)."""
        sims = np.where(sims > 0, sims, 0.0)

        if sims.shape[1] > self.n_neighbors:
            keep = np.argpartition(-sims, self.n_neighbors - 1, axis=1)[
                :, : self.n_neighbors
            ]
            trimmed = np.zeros_like(sims)
            np.put_along_axis(
                trimmed, keep, np.take_along_axis(sims, keep, axis=1), axis=1
            )
            sims = trimmed

        numerator = sims @ deviations
        denominator = sims.sum(axis=1)
        preds = np.where(
            denominator > 1e-9,
            fallback + numerator / np.maximum(denominator, 1e-9),
            fallback,
        )
        return np.clip(preds, config.RATING_MIN, config.RATING_MAX)

    def predict_new_user(
        self, rated: list[tuple[int, float]], film_ids: np.ndarray
    ) -> np.ndarray:
        """Modeldə olmayan yeni müştəri üçün proqnoz.

        Film-film oxşarlıq matrisi ONSUZ DA hazırdır və yeni müştəridən
        asılı deyil - ona görə modeli yenidən öyrətməyə ehtiyac yoxdur.
        Sadəcə yeni müştərinin verdiyi balları həmin matrisdə "gəzdiririk".
        """
        assert self.imap is not None and self.similarity is not None
        pairs = [
            (self.imap.film_pos[int(f)], float(r))
            for f, r in rated
            if int(f) in self.imap.film_pos
        ]
        film_ids = np.asarray(film_ids)
        if not pairs:
            return np.full(len(film_ids), self.global_mean)

        rated_pos = np.array([p for p, _ in pairs])
        values = np.array([r for _, r in pairs])
        user_mean = float(values.mean())
        deviations = values - user_mean  # öz ortasından kənara çıxma

        target_pos = np.array([self.imap.film_pos.get(int(f), -1) for f in film_ids])
        out = np.full(len(film_ids), user_mean, dtype=np.float64)
        known = target_pos >= 0
        if known.any():
            sims = self.similarity[np.ix_(target_pos[known], rated_pos)]
            out[known] = self._weighted_prediction(sims, deviations, user_mean)
        return np.clip(out, config.RATING_MIN, config.RATING_MAX)

    def recommend_new_user(
        self,
        rated: list[tuple[int, float]],
        candidates: np.ndarray,
        n: int = config.TOP_N,
    ) -> list[tuple[int, float]]:
        """Yeni müştəri üçün Top-N siyahısı."""
        candidates = np.asarray(candidates)
        if len(candidates) == 0:
            return []
        scores = self.predict_new_user(rated, candidates)
        order = np.lexsort((candidates, -scores))[:n]
        return [(int(candidates[i]), float(scores[i])) for i in order]

    # ------------------------------------------------------------------
    def similar_films(self, film_id: int, n: int = 5) -> list[tuple[int, float]]:
        """Verilmiş filmə ən oxşar N film: (film_id, oxşarlıq)."""
        assert self.imap is not None and self.similarity is not None
        i_pos = self.imap.film_pos.get(int(film_id))
        if i_pos is None:
            return []
        sims = self.similarity[i_pos]
        order = np.argsort(-sims)[:n]
        return [(int(self.imap.film_ids[j]), float(sims[j])) for j in order]
