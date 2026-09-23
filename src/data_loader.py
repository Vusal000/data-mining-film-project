"""
Verilənlərin oxunması
=====================

CSV faylları oxuyur və modellərin işləməsi üçün lazım olan kiçik
köməkçi strukturları qurur (id -> sətir/sütun nömrəsi uyğunluğu).

Niyə "indeks uyğunluğu" lazımdır?
    CSV-də film_id 1-dən 200-ə qədərdir, amma numpy matrisində sütunlar
    0-dan 199-a qədər nömrələnir. Bu iki nömrələmə arasında körpü qururuq.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# OXUMA FUNKSİYALARI
# --------------------------------------------------------------------------

def load_customers() -> pd.DataFrame:
    """customers.csv faylını oxuyur."""
    df = pd.read_csv(config.CUSTOMERS_CSV, encoding=config.CSV_ENCODING)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df


def load_films() -> pd.DataFrame:
    """films.csv faylını oxuyur və janrları siyahıya çevirir."""
    df = pd.read_csv(config.FILMS_CSV, encoding=config.CSV_ENCODING)
    # "Dram|Romantika" -> ["Dram", "Romantika"]
    df["genre_list"] = df["genres"].str.split("|")
    return df


def load_ratings() -> pd.DataFrame:
    """ratings.csv faylını oxuyur və zamana görə sıralayır."""
    df = pd.read_csv(config.RATINGS_CSV, encoding=config.CSV_ENCODING)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Üç faylı birlikdə oxuyur: (customers, films, ratings)."""
    return load_customers(), load_films(), load_ratings()


# --------------------------------------------------------------------------
# KÖMƏKÇİ STRUKTURLAR
# --------------------------------------------------------------------------

class IndexMap:
    """id <-> matris indeksi arasında çevirici.

    Nümunə:
        imap.film_pos[57]   -> 56   (film_id 57 matrisin 56-cı sütunudur)
        imap.film_ids[56]   -> 57   (əks istiqamət)
    """

    def __init__(self, customer_ids: np.ndarray, film_ids: np.ndarray):
        self.customer_ids = np.asarray(sorted(customer_ids))
        self.film_ids = np.asarray(sorted(film_ids))
        self.customer_pos = {int(c): i for i, c in enumerate(self.customer_ids)}
        self.film_pos = {int(f): i for i, f in enumerate(self.film_ids)}

    @property
    def n_customers(self) -> int:
        return len(self.customer_ids)

    @property
    def n_films(self) -> int:
        return len(self.film_ids)


def build_index(ratings: pd.DataFrame,
                all_customer_ids: np.ndarray | None = None,
                all_film_ids: np.ndarray | None = None) -> IndexMap:
    """Reytinq cədvəlindən IndexMap qurur.

    all_customer_ids / all_film_ids verilərsə, təlim setində heç reytinq
    almamış müştəri və filmlər də matrisə daxil edilir (sətir/sütun boş qalır).
    """
    cids = ratings["customer_id"].unique() if all_customer_ids is None else all_customer_ids
    fids = ratings["film_id"].unique() if all_film_ids is None else all_film_ids
    return IndexMap(cids, fids)


def build_matrix(ratings: pd.DataFrame, imap: IndexMap) -> np.ndarray:
    """Müştəri x film reytinq matrisi qurur. Boş xanalar 0 olur.

    0 "sıfır bal" demək deyil - "reytinq yoxdur" deməkdir. Modellər bunu
    ayrıca maska ilə nəzərə alır.
    """
    matrix = np.zeros((imap.n_customers, imap.n_films), dtype=np.float64)
    rows = ratings["customer_id"].map(imap.customer_pos).to_numpy()
    cols = ratings["film_id"].map(imap.film_pos).to_numpy()
    matrix[rows, cols] = ratings["rating"].to_numpy(dtype=np.float64)
    return matrix


def film_title_map(films: pd.DataFrame) -> dict[int, str]:
    """film_id -> film adı lüğəti (izahlarda istifadə olunur)."""
    return dict(zip(films["film_id"].astype(int), films["title"]))
