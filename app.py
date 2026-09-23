"""
Film Tövsiyə Sistemi - əsas tətbiq faylı
========================================

Bu fayl yalnız İKİ işi görür:
  1) Brauzer səhifəsinin ümumi ayarlarını qurur (başlıq, ikon, geniş görünüş)
  2) 8 səhifəni yan paneldə menyu kimi düzür

Hər səhifənin öz kodu pages/ qovluğundadır.

İŞƏ SALMAQ ÜÇÜN:
    streamlit run app.py
"""

import streamlit as st

# set_page_config MÜTLƏQ ilk Streamlit əmri olmalıdır
st.set_page_config(
    page_title="Film Tövsiyə Sistemi",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    st.Page("pages/0_ev.py", title="Ev", icon="🏠", default=True),
    st.Page("pages/1_verilenler.py", title="Verilənlər", icon="📊"),
    st.Page("pages/2_metodlar.py", title="Metodlar", icon="🧭"),
    st.Page("pages/3_muqayise.py", title="Model müqayisəsi", icon="🏆"),
    st.Page("pages/4_tovsiyeler.py", title="Müştəri tövsiyələri", icon="🎯"),
    st.Page("pages/5_soyuq_start.py", title="Soyuq start", icon="❄️"),
    st.Page("pages/6_arxitektura.py", title="Arxitektura", icon="🏗️"),
    st.Page("pages/7_neticeler.py", title="Nəticələr və məhdudiyyətlər", icon="📌"),
]

navigation = st.navigation(PAGES)
navigation.run()
