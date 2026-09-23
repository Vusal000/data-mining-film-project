"""
Bölgünün düzgünlüyü testləri
============================

Ən vacib sual: TEST verilənləri təsadüfən təlimə sızırmı?
Əgər sızırsa, bütün nəticələr etibarsızdır.
"""

import numpy as np

from src.split import check_no_leakage, split_ratings


def _keys(df):
    """Hər reytinqi unikal açara çevirir: (müştəri, film, zaman)."""
    return set(
        zip(
            df["customer_id"].astype(int),
            df["film_id"].astype(int),
            df["timestamp"].astype(str),
        )
    )


def test_butun_reytinqler_bir_defe_istifade_olunur(ratings, split):
    """train + val + test = ilkin verilənlərin hamısı, təkrarsız."""
    total = len(split.train) + len(split.val) + len(split.test)
    assert total == len(ratings)
    assert _keys(split.train) | _keys(split.val) | _keys(split.test) == _keys(ratings)


def test_destler_kesismir(split):
    """Heç bir reytinq eyni anda iki dəstdə ola bilməz."""
    train, val, test = _keys(split.train), _keys(split.val), _keys(split.test)
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_train_full_train_ve_val_cemidir(split):
    """train_full məhz train + val olmalıdır - testdən heç nə əlavə olunmur."""
    assert _keys(split.train_full) == _keys(split.train) | _keys(split.val)
    assert _keys(split.train_full) & _keys(split.test) == set()


def test_test_zamanca_en_sonuncudur(split):
    """Hər müştəri üçün test reytinqləri təlim reytinqlərindən SONRA olmalıdır."""
    train_full = split.train_full
    for cid, test_group in split.test.groupby("customer_id"):
        past = train_full.loc[train_full["customer_id"] == cid, "timestamp"]
        if len(past):
            assert test_group["timestamp"].min() >= past.max()


def test_val_trainden_sonra_gelir(split):
    """Validasiya train-dən sonra, test-dən əvvəl olmalıdır."""
    for cid, val_group in split.val.groupby("customer_id"):
        train_times = split.train.loc[split.train["customer_id"] == cid, "timestamp"]
        if len(train_times):
            assert val_group["timestamp"].min() >= train_times.max()


def test_her_musteri_uc_destde_de_var(ratings, split):
    """Hər müştərinin həm təlimi, həm testi olmalıdır (ən azı 15 reytinqi var)."""
    everyone = set(ratings["customer_id"].unique())
    assert set(split.train["customer_id"].unique()) == everyone
    assert set(split.test["customer_id"].unique()) == everyone


def test_test_payi_teqriben_20_faizdir(ratings, split):
    """Test payı 20%-ə yaxın olmalıdır."""
    share = len(split.test) / len(ratings)
    assert 0.15 <= share <= 0.25


def test_check_no_leakage_funksiyasi_hamisini_tesdiqleyir(split):
    """split.py-dakı öz yoxlama funksiyamız da bütün şərtləri təsdiqləməlidir."""
    assert all(check_no_leakage(split).values())


def test_bolgu_tekrarlanandir(ratings):
    """Eyni giriş -> eyni bölgü (təsadüfilik yoxdur)."""
    a = split_ratings(ratings)
    b = split_ratings(ratings)
    assert _keys(a.test) == _keys(b.test)
    assert _keys(a.val) == _keys(b.val)


def test_bolgu_zamana_gore_siralanib(split):
    """Hər dəstdə hər müştərinin reytinqləri zaman ardıcıllığındadır."""
    for part in (split.train, split.val, split.test):
        for _, group in part.groupby("customer_id"):
            times = group["timestamp"].to_numpy()
            assert np.all(times[:-1] <= times[1:])
