import streamlit as st
import pickle
import numpy as np
import pandas as pd
import utils
from tensorflow import keras


def show_train_predict_page_v1():
    st.title("Movie Recommendation")

    iteration_number = st.session_state.get("iteration_number", 100)
    feature_number = st.session_state.get("feature_number", 100)
    selected_optimizer = st.session_state.get("selected_optimizer", keras.optimizers.Adam(learning_rate=1e-1))

    st.subheader("""Choose Parameters""")
    st.write("You can change these parameters on Tune the Model page")
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

    # Call functions
    df_ratings, df_ratings_mean, df_movie = utils.read_data()
    num_users, num_movies, movieList = utils.get_info(df_ratings, df_ratings_mean)
    Y, R, my_ratings = utils.create_matrices(df_ratings, num_movies)

    # Prepare Dataset(select 30 most rated movies)
    df_ratings_mean_temp = df_ratings_mean.copy()
    all_genres_df, list_genre = utils.prepare_selected_movies(df_ratings_mean_temp)

    randomstate_user = st.slider('Give me a random number', min_value=1, max_value=100, value=42, step=1)
    movienumber_user = st.slider('How many movies do you want to rate', min_value=6, max_value=20, value=6, step=1)
    st.session_state["randomstate"] = randomstate_user
    st.session_state["movienumber"] = movienumber_user
    randomstate = st.session_state.get("randomstate", 42)
    movienumber = st.session_state.get("movienumber", 6)

    # Create checkboxes for each genre
    selected_genres_user = st.multiselect('Select Genres', list_genre)
    selected_genres = [str(genre) for genre in selected_genres_user]
    # selected_genres = ['Action']
    # st.write(list(selected_genres))

    # Get highest number of rated movies
    all_genres_df_2 = all_genres_df.sort_values(by="number_of_ratings", ascending=False)
    st.write('Length of movie database:', len(all_genres_df_2))
    st.write('Sample from DB:', all_genres_df_2.head())

    if not selected_genres:
        all_genres_df_3 = all_genres_df_2.copy().head(100)
        all_genres_df_3 = all_genres_df_3.sample(len(all_genres_df_3), random_state=randomstate)
    else:
        # Selected movies for all genre types (Most rated 30 movies for each genre)
        selected_genres_dflist = []
        for genre in selected_genres:
            print(genre)
            genre_temp = all_genres_df_2[np.array(all_genres_df_2.filter(regex=genre) == 1).reshape
            (len(all_genres_df_2), )].sort_values(
                by='number_of_ratings', ascending=False).head(30)
            print(genre_temp)
            # remove already selected movies
            genre_temp_movie_id = list(genre_temp["movie_id_2"])
            print(genre_temp_movie_id)
            all_genres_df_2 = all_genres_df_2[~all_genres_df_2.movie_id_2.isin(genre_temp_movie_id)]

            selected_genres_dflist.append(genre_temp)
            print(selected_genres_dflist)

        all_genres_df_3 = pd.concat(selected_genres_dflist)
        all_genres_df_3 = all_genres_df_3.sample(len(all_genres_df_3), random_state=randomstate)

        st.write('Length of selected genres:', len(all_genres_df_3))
        st.write('Sample from selected genres', all_genres_df_3.head())

    st.subheader("""Click checkbox to enter your own ratings.""")
    st.write("Give 0, if you haven't seen the movie yet. Give ratings from 1 to 5.")

    checkbox_b = st.checkbox('Show me movies')
    if checkbox_b:
        # Select movies based on genres
        # selected_movies = all_genres_df_3.sample(n=movienumber, random_state=randomstate)
        selected_movies = all_genres_df_3.head(movienumber)
        for i in range(movienumber):
            my_ratings, all_genres_df_3 = utils.get_ratings_from_user_2(movieList, i, selected_movies, my_ratings,
                                                                        all_genres_df_3)

    train_button = st.button("Recommend movies!")
    if train_button:
        st.subheader('Thank you. Wait for the recommendation!')
        st.write('Your data is added. The model is being retrained to give you personal recommendations.')

        st.write('\n\nYour Original ratings:\n')
        for i in range(len(my_ratings)):
            if my_ratings[i] > 0:
                st.write(f'Original {my_ratings[i]}, for {movieList[i]}')

        my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
        Y = np.c_[my_ratings, Y]  # Add new user ratings to Y
        R = np.c_[(my_ratings != 0).astype(int), R]  # Add new user indicator matrix to R
        Ynorm, Ymean = utils.normalizeRatings(Y, R)  # Normalize the Dataset
        W, X, b = utils.train_data(Y, Ynorm, R, selected_optimizer, iteration_number, feature_number)  # TRAIN

        my_predictions = utils.prediction(W, X, b, Ymean, my_ratings, movieList)

        st.subheader('Our recommendations for you! Enjoy!')
        utils.give_recommendation(my_predictions, my_rated, movieList, df_ratings_mean)

        # analysis_button = st.checkbox("Show All Analysis")
        #
        # # ALL ANALYSIS
        # if analysis_button:
        #
        #     st.write("""#### Selected Parameters""")
        #     st.write(f"iteration number: {iteration_number}")
        #     st.write(f"feature number: {feature_number}")
        #     if "adam" in str(selected_optimizer).lower():
        #         st.write(f"selected_optimizer: {'Adam'}")
        #     elif "sgd" in str(selected_optimizer).lower():
        #         st.write(f"selected_optimizer: {'SGD'}")
        #     elif "rmsprop" in str(selected_optimizer).lower():
        #         st.write(f"selected_optimizer: {'RMSProp'}")
        #     else:
        #         st.write("Please choose optimizer in Tune Model Page.")
        #     st.write("You can change these parameters on Tune the Model page")
        #
        #     st.write('Length of movie database:', len(all_genres_df_2))
        #     st.write('Sample from DB:', all_genres_df_2.head())
        #
        #     st.write('Length of selected genres:', len(all_genres_df_3))
        #     st.write('Sample from selected genres', all_genres_df_3.head())
        #
        #     st.write('\n\nYour Original ratings:\n')
        #     for i in range(len(my_ratings)):
        #         if my_ratings[i] > 0:
        #             st.write(f'Original {my_ratings[i]}, for {movieList[i]}')
        #
        #     st.write('These are the predictions for your own ratings.')
        #     for i in range(len(my_ratings)):
        #         if my_ratings[i] > 0:
        #             st.write(
        #                 f'{movieList[i]}: Your rating is {my_ratings[i]}, Predicted rating is {my_predictions[i]:0.2f}')
