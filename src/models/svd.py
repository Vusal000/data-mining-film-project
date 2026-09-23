"""
Model 3: SVD matris faktorizasiyası (Funk SVD, meyilli / biased)
================================================================

GÜNDƏLİK BƏNZƏTMƏ:
    Hər filmi və hər müştərini eyni "gizli xüsusiyyətlər" ölçüsündə
    təsvir edirik. Məsələn (model bunları özü tapır, biz ad vermirik):
        1-ci ölçü: "nə qədər döyüşlüdür / sakitdir"
        2-ci ölçü: "nə qədər ciddi / əyləncəlidir"
    Müştərinin vektoru onun bu ölçülərə marağını, filmin vektoru isə
    filmin həmin ölçülərdəki dozasını göstərir. İki vektorun skalyar
    hasili nə qədər böyükdürsə, uyğunluq bir o qədər yaxşıdır.

PROQNOZ DÜSTURU:

    proqnoz(u,i) = mu + b_u + b_i + (p_u . q_i)

        mu   - bütün reytinqlərin ümumi ortası
        b_u  - müştərinin meyli (səxavətli / sərt qiymətləndirən)
        b_i  - filmin meyli (ümumiyyətlə yaxşı / zəif film)
        p_u  - müştərinin k ölçülü gizli vektoru
        q_i  - filmin k ölçülü gizli vektoru

NECƏ ÖYRƏNİR? (SGD - stokastik qradiyent enişi)
    Hər mövcud reytinqə bir-bir baxırıq:
        xəta = həqiqi bal - proqnoz
    və hər parametri xətanı azaldacaq istiqamətdə KİÇİK bir addım dəyişirik.
    Bu, 20 dəfə (epoch) təkrarlanır.

        b_u <- b_u + lr * (xəta - reg * b_u)
        p_u <- p_u + lr * (xəta * q_i - reg * p_u)
        q_i <- q_i + lr * (xəta * p_u - reg * q_i)

    lr  = 0.005  (addım ölçüsü)
    reg = 0.02   (requlyarizasiya: parametrlərin həddən artıq böyüməsinin
                  qarşısını alır, yəni əzbərləməyə mane olur)

YENİ İSTİFADƏÇİ (fold-in):
    Film vektorları (q) DONDURULUR, yalnız yeni müştərinin b_u və p_u
    vektoru öyrədilir. Bütün modeli yenidən öyrətməyə ehtiyac qalmır -
    soyuq start səhifəsində canlı tövsiyə buna görə mümkündür.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..data_loader import IndexMap, build_index
from .base import BaseRecommender


class SVDRecommender(BaseRecommender):
    """Öz əlimizlə numpy ilə yazılmış meyilli SVD (Funk SVD)."""

    key = "svd"

    def __init__(
        self,
        n_factors: int = config.SVD_DEFAULT_FACTORS,
        n_epochs: int = config.SVD_EPOCHS,
        lr: float = config.SVD_LR,
        reg: float = config.SVD_REG,
        seed: int = config.SEED,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

        self.imap: IndexMap | None = None
        self.b_u: np.ndarray | None = None
        self.b_i: np.ndarray | None = None
        self.P: np.ndarray | None = None  # müştəri vektorları (müştəri x k)
        self.Q: np.ndarray | None = None  # film vektorları (film x k)
        self.train_rmse_history: list[float] = []
        self.user_rated: dict[int, list[tuple[int, float]]] = {}

    # ------------------------------------------------------------------
    def fit(self, train: pd.DataFrame,
            all_film_ids: np.ndarray | None = None) -> "SVDRecommender":
        rng = np.random.default_rng(self.seed)

        self.global_mean = float(train["rating"].mean())
        self.imap = build_index(train, all_film_ids=all_film_ids)

        n_u, n_i = self.imap.n_customers, self.imap.n_films

        # Başlanğıc dəyərlər: meyillər 0, gizli vektorlar kiçik təsadüfi ədədlər
        self.b_u = np.zeros(n_u)
        self.b_i = np.zeros(n_i)
        self.P = rng.normal(0.0, 0.10, size=(n_u, self.n_factors))
        self.Q = rng.normal(0.0, 0.10, size=(n_i, self.n_factors))

        users = train["customer_id"].map(self.imap.customer_pos).to_numpy()
        items = train["film_id"].map(self.imap.film_pos).to_numpy()
        ratings = train["rating"].to_numpy(dtype=np.float64)
        n = len(ratings)

        # İzah funksiyası üçün: hər müştərinin təlimdə verdiyi ballar
        self.user_rated = {}
        for cid, group in train.groupby("customer_id"):
            self.user_rated[int(cid)] = list(
                zip(group["film_id"].astype(int), group["rating"].astype(float))
            )

        self.train_rmse_history = []

        # Sürət üçün yerli dəyişənlər (məntiq dəyişmir, sadəcə axtarış azalır)
        P, Q, b_u, b_i = self.P, self.Q, self.b_u, self.b_i
        lr, reg, mu = self.lr, self.reg, self.global_mean
        users_list = users.tolist()
        items_list = items.tolist()
        ratings_list = ratings.tolist()

        # --------------------- TƏLİM DÖVRÜ ---------------------
        for _ in range(self.n_epochs):
            order = rng.permutation(n).tolist()  # hər epoch-da qarışdırırıq
            squared_error = 0.0

            for idx in order:
                u = users_list[idx]
                i = items_list[idx]

                p_u = P[u]  # P-nin sətrinə "baxış" (view) - dəyişiklik birbaşa düşür
                q_i = Q[i]

                pred = mu + b_u[u] + b_i[i] + float(p_u @ q_i)
                err = ratings_list[idx] - pred
                squared_error += err * err

                # Meyillərin yenilənməsi
                b_u[u] += lr * (err - reg * b_u[u])
                b_i[i] += lr * (err - reg * b_i[i])

                # Gizli vektorların yenilənməsi.
                # p_u-nun KÖHNƏ dəyəri saxlanılır, çünki q_i yenilənərkən
                # artıq dəyişmiş p_u deyil, köhnəsi istifadə olunmalıdır.
                p_old = p_u.copy()
                p_u += lr * (err * q_i - reg * p_old)
                q_i += lr * (err * p_old - reg * q_i)

            self.train_rmse_history.append(float(np.sqrt(squared_error / n)))

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def _raw_predict(self, u_pos: int | None, i_pos: int | None) -> float:
        value = self.global_mean
        if u_pos is not None:
            value += self.b_u[u_pos]
        if i_pos is not None:
            value += self.b_i[i_pos]
        if u_pos is not None and i_pos is not None:
            value += float(self.P[u_pos] @ self.Q[i_pos])
        return value

    def predict(self, customer_id: int, film_id: int) -> float:
        assert self.imap is not None
        u_pos = self.imap.customer_pos.get(int(customer_id))
        i_pos = self.imap.film_pos.get(int(film_id))
        return self._clip(self._raw_predict(u_pos, i_pos))

    def predict_many(self, customer_ids, film_ids) -> np.ndarray:
        """Vektorlaşdırılmış proqnoz - bütün cütlər bir anda."""
        assert self.imap is not None
        u_pos = np.array(
            [self.imap.customer_pos.get(int(c), -1) for c in customer_ids]
        )
        i_pos = np.array([self.imap.film_pos.get(int(f), -1) for f in film_ids])

        out = np.full(len(u_pos), self.global_mean, dtype=np.float64)
        has_u, has_i = u_pos >= 0, i_pos >= 0

        out[has_u] += self.b_u[u_pos[has_u]]
        out[has_i] += self.b_i[i_pos[has_i]]
        both = has_u & has_i
        if both.any():
            out[both] += np.einsum(
                "ij,ij->i", self.P[u_pos[both]], self.Q[i_pos[both]]
            )
        return np.clip(out, config.RATING_MIN, config.RATING_MAX)

    # ------------------------------------------------------------------
    def explain(self, customer_id: int, film_id: int, titles: dict[int, str]) -> str:
        """Müştərinin bəyəndiyi, gizli vektoru bu filmə ən yaxın olan filmləri tapır."""
        assert self.imap is not None
        i_pos = self.imap.film_pos.get(int(film_id))
        history = self.user_rated.get(int(customer_id), [])
        liked = [(f, r) for f, r in history if r >= config.LIKE_THRESHOLD]

        if i_pos is None or not liked:
            return "Ümumi meyllərə əsaslanan tövsiyə."

        target = self.Q[i_pos]
        target_norm = np.linalg.norm(target)
        scored: list[tuple[float, int]] = []
        for fid, _ in liked:
            j = self.imap.film_pos.get(int(fid))
            if j is None:
                continue
            denom = target_norm * np.linalg.norm(self.Q[j])
            if denom > 0:
                scored.append((float(target @ self.Q[j] / denom), int(fid)))

        if not scored:
            return "Ümumi meyllərə əsaslanan tövsiyə."

        scored.sort(reverse=True)
        names = [titles.get(fid, f"#{fid}") for _, fid in scored[:2]]
        joined = "» və «".join(names)
        return (
            f"«{joined}» filmlərini bəyəndiyiniz üçün - bu film onlarla eyni "
            f"gizli xüsusiyyətləri paylaşır."
        )

    # ------------------------------------------------------------------
    # YENİ İSTİFADƏÇİ ÜÇÜN FOLD-IN
    # ------------------------------------------------------------------
    def fold_in(
        self,
        rated: list[tuple[int, float]],
        n_epochs: int = config.SVD_FOLDIN_EPOCHS,
    ) -> tuple[float, np.ndarray]:
        """Yeni müştəri üçün yalnız b_u və p_u öyrədir (film vektorları sabitdir).

        rated: [(film_id, bal), ...]
        Qaytarır: (b_u, p_u)
        """
        assert self.imap is not None
        rng = np.random.default_rng(self.seed)
        b_new = 0.0
        p_new = rng.normal(0.0, 0.10, size=self.n_factors)

        pairs = [
            (self.imap.film_pos[int(f)], float(r))
            for f, r in rated
            if int(f) in self.imap.film_pos
        ]
        if not pairs:
            return 0.0, np.zeros(self.n_factors)

        for _ in range(n_epochs):
            for i_pos, r in pairs:
                pred = self.global_mean + b_new + self.b_i[i_pos] + p_new @ self.Q[i_pos]
                err = r - pred
                b_new += self.lr * (err - self.reg * b_new)
                p_new += self.lr * (err * self.Q[i_pos] - self.reg * p_new)

        return float(b_new), p_new

    def predict_foldin(
        self, b_new: float, p_new: np.ndarray, film_ids: np.ndarray
    ) -> np.ndarray:
        """Fold-in ilə alınmış vektorla bal proqnozu."""
        assert self.imap is not None
        positions = np.array([self.imap.film_pos.get(int(f), -1) for f in film_ids])
        out = np.full(len(positions), self.global_mean + b_new, dtype=np.float64)
        known = positions >= 0
        if known.any():
            out[known] += self.b_i[positions[known]]
            out[known] += self.Q[positions[known]] @ p_new
        return np.clip(out, config.RATING_MIN, config.RATING_MAX)

    def recommend_foldin(
        self,
        rated: list[tuple[int, float]],
        candidates: np.ndarray,
        n: int = config.TOP_N,
    ) -> list[tuple[int, float]]:
        """Yeni müştəri üçün tövsiyə siyahısı."""
        b_new, p_new = self.fold_in(rated)
        scores = self.predict_foldin(b_new, p_new, candidates)
        order = np.lexsort((candidates, -scores))[:n]
        return [(int(candidates[i]), float(scores[i])) for i in order]
