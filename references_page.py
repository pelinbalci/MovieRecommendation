import streamlit as st


def show_references():
    st.title("References")

    st.write("""This notebook is prepared by the notes from Unsupervised Learning, Recommenders, 
    Reinforcement Learning by DeepLearning.AI Coursera. I would like to thank Andrew NG for this great lecture. 
    The training module and vectorized cost function are directly copied from the lecture. I prepared the data from scratch, 
    get input data, made the code modular (app.py, train_predict_page.py and utils.py). Also, you can run the project on a 
    dashboard which is prepared by streamlit:)""")


    st.write("""The data set is derived from the 
    [MovieLens "ml-latest-small"](https://grouplens.org/datasets/movielens/latest/) dataset.   
    [F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on 
    Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>]""")

    st.write("""I would like to thank all the mentors in the streamlit discussion forum and Patrick Loeber for the great 
    tutorial about Streamlit. You may find the related GitHub page here: https://github.com/patrickloeber/ml-app-salaryprediction""")
