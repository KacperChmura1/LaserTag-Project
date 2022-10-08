from explore_page import show_explore_page
import streamlit as st

from predict_page import show_predict_page
from ann_page import show_ann_page
#Pages choose
page = st.sidebar.selectbox("Explore Or Predict", {"Explore","ML Predict","ANN Predict"})
if page == "ML Predict":
    show_predict_page()
elif page == "Explore":
    show_explore_page()
else:
    show_ann_page()

