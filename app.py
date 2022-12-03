"""
The original source: https://github.com/patrickloeber/ml-app-salaryprediction

"""

import streamlit as st
from train_predict_page import show_train_predict_page
# from explore_page import show_explore_page


page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_train_predict_page()
# else:
#     show_explore_page()