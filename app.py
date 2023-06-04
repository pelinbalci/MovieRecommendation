import streamlit as st
from train_predict_page_v0 import show_train_predict_page_v0
from explore_page import show_explore_page
from references_page import show_references
from tuning_page import show_tuning_page
from train_predict_page_v1 import show_train_predict_page_v1

page = st.sidebar.selectbox("Pages", ("Recommendation_v0", "Recommendation_v1", "Tune the Model", "Explore", "References"))

if page == "Recommendation_v0":
    show_train_predict_page_v0()

elif page == "Recommendation_v1":
    show_train_predict_page_v1()

elif page == "Tune the Model":
    show_tuning_page()

elif page == "Explore":
    show_explore_page()

else:
    show_references()
