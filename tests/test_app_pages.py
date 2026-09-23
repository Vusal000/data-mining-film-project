"""
Tətbiq səhifələrinin testi
==========================

Hər səhifəni brauzersiz (headless) işlədir və <b>heç bir xəta olmadığını</b>
yoxlayır. Streamlit-in öz test aləti `AppTest` bunun üçündür.

Niyə lazımdır? Səhifədə kiçik bir yazı səhvi (məsələn səhv sütun adı)
yalnız həmin səhifə açılanda üzə çıxır. Bu test onları əvvəlcədən tutur.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = sorted(
    p.name for p in (Path(__file__).resolve().parents[1] / "pages").glob("*.py")
)


@pytest.mark.parametrize("page_name", PAGES)
def test_sehife_xetasiz_acilir(page_name):
    """Səhifə sonuna qədər işləməli və heç bir exception atmamalıdır."""
    app = AppTest.from_file(f"pages/{page_name}", default_timeout=300)
    app.run()

    assert not app.exception, (
        f"{page_name} səhifəsində xəta: "
        + "; ".join(str(e.value) for e in app.exception)
    )
    # Streamlit-in st.error bloku da problem əlamətidir
    assert not app.error, (
        f"{page_name} səhifəsində st.error göstərildi: "
        + "; ".join(str(e.value) for e in app.error)
    )


@pytest.mark.parametrize("page_name", PAGES)
def test_html_teqleri_xam_gorunmur(page_name):
    """REGRESSİYA TESTİ.

    st.success / st.info / st.warning / st.caption bloklari markdown başa
    düşür, HTML-i YOX. Əgər ora səhvən <b>...</b> yazılsa, istifadəçi
    ekranda hərfi olaraq "<b>" görür. Bu test bunun qarşısını alır.
    """
    app = AppTest.from_file(f"pages/{page_name}", default_timeout=300)
    app.run()

    html_like = ("<b>", "</b>", "<br>", "<span", "<div", "<i>")
    plain_blocks = (
        list(app.success)
        + list(app.info)
        + list(app.warning)
        + list(app.error)
        + list(app.caption)
    )
    for block in plain_blocks:
        text = str(block.value)
        found = [tag for tag in html_like if tag in text]
        assert not found, (
            f"{page_name}: markdown blokunda xam HTML teqi var {found} → "
            f"{text[:120]}"
        )


def test_soyuq_start_canli_tovsiye_verir():
    """Soyuq start səhifəsində 3 filmə ulduz verdikdə tövsiyə görünməlidir.

    Bu, səhifənin ƏSAS interaktiv axınıdır: yeni istifadəçi → fold-in →
    canlı tövsiyə. Yalnız səhifənin açılmasını yoxlamaq kifayət deyil.
    """
    app = AppTest.from_file("pages/5_soyuq_start.py", default_timeout=300)
    app.run()

    # Ulduz seçiciləri (hər film üçün bir selectbox)
    assert len(app.selectbox) >= 3, "Ulduz seçiciləri tapılmadı"

    for box, stars in zip(app.selectbox[:3], (5, 4, 5)):
        box.set_value(stars)
    app.run()

    assert not app.exception, "; ".join(str(e.value) for e in app.exception)

    # "3 reytinq qəbul edildi" mesajı görünməlidir
    success_text = " ".join(str(block.value) for block in app.success)
    assert "reytinq qəbul edildi" in success_text, (
        f"Tövsiyə bloku açılmadı. Görünən success mesajları: {success_text[:200]}"
    )


def test_butun_sehifeler_movcuddur():
    """app.py-da elan olunan 8 səhifənin hamısı faylda mövcud olmalıdır."""
    assert len(PAGES) == 8, f"Gözlənilən 8 səhifə, tapıldı: {PAGES}"
