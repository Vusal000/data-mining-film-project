"""
pytest üçün ümumi hazırlıq
==========================

Bu fayl layihənin kök qovluğunda olduğu üçün pytest `src` paketini
tapa bilir. Burada həm də bütün testlərin paylaşdığı verilənlər var -
beləliklə CSV faylları hər test üçün yenidən oxunmur.
"""

import pytest

from src.data_loader import load_all
from src.split import split_ratings


@pytest.fixture(scope="session")
def data():
    """(customers, films, ratings) - bir dəfə oxunur."""
    return load_all()


@pytest.fixture(scope="session")
def ratings(data):
    return data[2]


@pytest.fixture(scope="session")
def films(data):
    return data[1]


@pytest.fixture(scope="session")
def split(ratings):
    """train / val / test bölgüsü - bir dəfə hesablanır."""
    return split_ratings(ratings)
