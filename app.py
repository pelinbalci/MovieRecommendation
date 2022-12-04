import streamlit as st
from train_predict_page import show_train_predict_page
from explore_page import show_explore_page
from references_page import show_references

page = st.sidebar.selectbox("Pages", ("Recommendation", "Explore", "References"))

if page == "Recommendation":
    show_train_predict_page()
elif page == "Explore":
    show_explore_page()
else:
    show_references()
