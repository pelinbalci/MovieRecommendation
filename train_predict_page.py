import streamlit as st
import pickle
import numpy as np
import pandas as pd
import utils
from tensorflow import keras

from tuning_page import show_tuning_page


def show_train_predict_page():
    st.title("Movie Recommendation")
    # st.write("""### Let's start with model training hyper-parameters!""")
    # st.write("You can simply enter the default values:) ")

    # iteration_number = st.number_input("Number of Iterations (default: 100)", min_value=10, max_value=100, step=10)
    # feature_number = st.number_input("Number of Movie Features (default: 100)", min_value=10, max_value=100, step=10)
    # opt_select = st.radio("Optimization Type (default: Adam)", ("Adam", "SGD", "Less Known"))
    #
    # if opt_select == "Adam":
    #     selected_optimizer = keras.optimizers.Adam(learning_rate=1e-1)
    # elif opt_select == 'SGD':
    #     selected_optimizer = keras.optimizers.SGD(learning_rate=1e-1)
    # else:
    #     selected_optimizer = keras.optimizers.RMSprop(learning_rate=1e-1)

    # param_dict = show_tuning_page()
    # iteration_number = param_dict.get("iteration_number", 100)
    # feature_number = param_dict.get("feature_number", 100)
    # selected_optimizer = param_dict.get("selected_optimizer", "Adam")

    iteration_number = st.session_state.get("iteration_number", 100)
    feature_number = st.session_state.get("feature_number", 100)
    selected_optimizer = st.session_state.get("selected_optimizer", keras.optimizers.Adam(learning_rate=1e-1))

    st.write("""#### Selected Parameters""")
    st.write(f"iteration number: {iteration_number}")
    st.write(f"feature number: {feature_number}")
    if "adam" in str(selected_optimizer).lower():
        st.write(f"selected_optimizer: {'Adam'}")
    elif "sgd" in str(selected_optimizer).lower():
        st.write(f"selected_optimizer: {'SGD'}")
    elif "rmsprop" in str(selected_optimizer).lower():
        st.write(f"selected_optimizer: {'RMSProp'}")
    else:
        st.write("Please choose optimizer in Tune Model Page.")
    st.write("You can change these parameters on Tune the Model page")

    # Call functions
    df_ratings, df_ratings_mean, df_movie = utils.read_data()
    num_users, num_movies, movieList = utils.get_info(df_ratings, df_ratings_mean)
    Y, R, my_ratings = utils.create_matrices(df_ratings, num_movies)

    # Prapre Dataset(select 30 most rated movies)
    df_ratings_mean_temp = df_ratings_mean.copy()
    all_genres_df = utils.prepare_selected_movies(df_ratings_mean_temp)
    # all_genres_df_2 = all_genres_df.sample(frac=1)

    if "movie_order" not in st.session_state:
        st.session_state["movie_order"] = list(all_genres_df["movie_id_2"])

    st.write("""#### It is time to enter your own ratings!""")
    st.write("Give 0, if you haven't seen the movie yet. Give ratings from 1 to 5.")

    selection = st.radio("Select Movies based on: ", ("Most Rated", "Highest Rated", "Less Known"))
    if selection == "Most Rated":
        all_genres_df_temp_1 = all_genres_df.sort_values(by="number_of_ratings", ascending=False)
        all_genres_df_temp_index_1 = list(all_genres_df_temp_1["movie_id_2"])
        st.session_state.movie_order = all_genres_df_temp_index_1
        print(all_genres_df_temp_index_1)
    elif selection == "Highest Rated":
        all_genres_df_temp_2 = all_genres_df.sort_values(by="mean_rating", ascending=False)
        all_genres_df_temp_index_2 = list(all_genres_df_temp_2["movie_id_2"])
        st.session_state.movie_order = all_genres_df_temp_index_2
        print(all_genres_df_temp_index_2)
    else:
        all_genres_df_temp_3 = all_genres_df.sort_values(by="number_of_ratings", ascending=True)
        all_genres_df_temp_index_3 = list(all_genres_df_temp_3["movie_id_2"])
        st.session_state.movie_order = all_genres_df_temp_index_3
        print(all_genres_df_temp_index_3)

    all_genres_df_2 = all_genres_df.reindex(st.session_state["movie_order"])

    # all_genres_df_2 = all_genres_df.copy()
    checkbox_b = st.checkbox('Rate comedies?')
    if checkbox_b:
        # all_genres_df_2 = all_genres_df.sample(frac=1)
        # filter dataframe based on genre
        selected_movies = all_genres_df_2[all_genres_df_2['genres'].str.contains("Comedy")]
        for i in range(3):
            my_ratings, all_genres_df_2 = utils.get_ratings_from_user_2(movieList, i, selected_movies, my_ratings,
                                                                        all_genres_df_2)

    checkbox_b = st.checkbox('Rate sci-fi?')
    if checkbox_b:
        # all_genres_df_2 = all_genres_df.sample(frac=1)
        # filter dataframe based on genre
        selected_movies = all_genres_df_2[all_genres_df_2['genres'].str.contains("Sci-Fi")]
        for i in range(3):
            my_ratings, all_genres_df_2 = utils.get_ratings_from_user_2(movieList, i, selected_movies, my_ratings,
                                                                        all_genres_df_2)

    checkbox_b = st.checkbox('Rate romance?')
    if checkbox_b:
        # all_genres_df_2 = all_genres_df.sample(frac=1)
        # filter dataframe based on genre
        selected_movies = all_genres_df_2[all_genres_df_2['genres'].str.contains("Romance")]
        for i in range(3):
            my_ratings, all_genres_df_2 = utils.get_ratings_from_user_2(movieList, i, selected_movies, my_ratings,
                                                                        all_genres_df_2)

    st.write('Thank you:) Wait for the recommendation!')
    st.write('\n\nOriginal vs Predicted ratings:\n')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            st.write(f'Original {my_ratings[i]}, for {movieList[i]}')

    train_button = st.button("Recommend movies!")
    if train_button:
        st.subheader('Your data is added. Retraining to give you personal recommendations:) ')
        my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
        Y = np.c_[my_ratings, Y]  # Add new user ratings to Y
        R = np.c_[(my_ratings != 0).astype(int), R]  # Add new user indicator matrix to R
        Ynorm, Ymean = utils.normalizeRatings(Y, R)  # Normalize the Dataset
        W, X, b = utils.train_data(Y, Ynorm, R, selected_optimizer, iteration_number, feature_number)  # TRAIN
        st.subheader('These are the predictions for your own ratings.')
        my_predictions = utils.prediction(W, X, b, Ymean, my_ratings, movieList)
        st.subheader('Recommended movies! Enjoy!')
        utils.give_recommendation(my_predictions, my_rated, movieList)
