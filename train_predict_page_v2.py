import streamlit as st
import pickle
import numpy as np
import pandas as pd
import utils
from tensorflow import keras


def show_train_predict_page_v2():
    st.title("Personalized Movie Recommendation")
    st.subheader("with TensorFlow and Content Based Filtering")
    st.subheader(" ")
    st.subheader("How to Use?")
    st.write("1. Change or Don't change the magic number. It will change the movies you rate.")
    st.write("2. You need to bring your own ratings. Select the number of movies you want to rate.")
    st.write("3. Select genre for recommendation")
    st.write("4. Click check box")
    st.write("5. Press Recommend Movies button")
    st.subheader(" ")

    iteration_number = st.session_state.get("iteration_number", 100)
    feature_number = st.session_state.get("feature_number", 100)
    selected_optimizer = st.session_state.get("selected_optimizer", keras.optimizers.Adam(learning_rate=1e-1))

    # Call functions
    df_ratings, df_ratings_mean, df_movie = utils.read_data()
    num_users, num_movies, movieList = utils.get_info(df_ratings, df_ratings_mean)
    Y, R, my_ratings = utils.create_matrices(df_ratings, num_movies)

    # Prepare Dataset(select 30 most rated movies)
    df_ratings_mean_temp = df_ratings_mean.copy()
    all_genres_df, list_genre = utils.prepare_selected_movies(df_ratings_mean_temp)

    randomstate_user = st.slider('MAGIC NUMBER FOR YOUR RATINGS', min_value=1, max_value=100, value=42, step=1)
    movienumber_user = st.slider('NUMBER OF MOVIES U WANT TO RATE', min_value=6, max_value=20, value=6, step=1)
    st.session_state["randomstate"] = randomstate_user
    st.session_state["movienumber"] = movienumber_user
    randomstate = st.session_state.get("randomstate", 42)
    movienumber = st.session_state.get("movienumber", 6)

    # Create checkboxes for each genre
    selected_genres_user = st.multiselect('SELECT GENRES', list_genre)
    selected_genres = [str(genre) for genre in selected_genres_user]
    st.session_state['selected_genre'] = selected_genres

    # Get highest number of rated movies
    all_genres_df_2 = all_genres_df.sort_values(by="number_of_ratings", ascending=False)
    # st.write('Length of movie database:', len(all_genres_df_2))

    if not selected_genres:
        all_genres_df_3 = all_genres_df_2.copy().head(100)
        all_genres_df_3 = all_genres_df_3.sample(len(all_genres_df_3), random_state=randomstate)
    else:
        # Selected movies for all genre types (Most rated 30 movies for each genre)
        all_genres_df_3 = utils.filter_genre(selected_genres, all_genres_df_2)
        all_genres_df_3 = all_genres_df_3.sample(len(all_genres_df_3), random_state=randomstate)

        # st.write('Length of selected genres:', len(all_genres_df_3))

    checkbox_b = st.checkbox("""CLICK to RATE MOVIES for SELECTED GENRES! """)
    st.write("Give 0, if you haven't seen the movie yet.")

    if checkbox_b:
        # Select movies based on genres
        # selected_movies = all_genres_df_3.sample(n=movienumber, random_state=randomstate)
        selected_movies = all_genres_df_3.head(movienumber)
        for i in range(movienumber):
            my_ratings, all_genres_df_3 = utils.get_ratings_from_user_2(movieList, i, selected_movies, my_ratings,
                                                                        all_genres_df_3)
        st.write("If you don't like these movies, change the MAGIC NUMBER!")

    train_button = st.button("RECOMMEND MOVIES!")
    st.write("P.S. You can change the hyper-parameters on 'Tune the Model' page")
    if train_button:
        st.subheader('Thank you. Wait for the recommendation!')
        st.write('The model is being retrained to give you personal recommendations.')

        my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
        Y = np.c_[my_ratings, Y]  # Add new user ratings to Y
        R = np.c_[(my_ratings != 0).astype(int), R]  # Add new user indicator matrix to R
        Ynorm, Ymean = utils.normalizeRatings(Y, R)  # Normalize the Dataset
        W, X, b = utils.train_data(Y, Ynorm, R, selected_optimizer, iteration_number, feature_number)  # TRAIN

        my_predictions = utils.prediction(W, X, b, Ymean, my_ratings, movieList)

        st.subheader('Our recommendations for you! Enjoy!')
        utils.give_recommendation(my_predictions, my_rated, movieList, all_genres_df_2)
