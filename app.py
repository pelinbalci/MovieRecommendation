import streamlit as st

from explore_page import show_explore_page
from references_page import show_references
from tuning_page import show_tuning_page
from tutorial import show_tutorial
from train_predict_page_v0 import show_train_predict_page_v0
from train_predict_page_v1 import show_train_predict_page_v1
from train_predict_page_v2 import show_train_predict_page_v2

page = st.sidebar.selectbox("Pages", ("Recommendation_v2",
                                      #"Recommendation_v1",
                                      # "Recommendation_v0",
                                      "Tune the Model",
                                      "Explore", "Tutorial", "References"))

if page == "Recommendation_v2":
    show_train_predict_page_v2()

# elif page == "Recommendation_v1":
#     show_train_predict_page_v1()
#
# elif page == "Recommendation_v0":
#     show_train_predict_page_v0()

elif page == "Tune the Model":
    show_tuning_page()

elif page == "Explore":
    show_explore_page()

elif page == "Tutorial":
    show_tutorial()
else:
    show_references()
