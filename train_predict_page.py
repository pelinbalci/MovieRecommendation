"""
The original source: https://github.com/patrickloeber/ml-app-salaryprediction

"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import utils


def show_train_predict_page():
    st.title("Movie Recommendation")
    st.write("""### Let's start with your own ratings!  :)""")
    st.write("Give 0, if you haven't seen the movie yet. Give ratings from 1 to 5.")

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
    comedy_b = st.checkbox('Rate comedies?')
    if comedy_b:
        # all_genres_df_2 = all_genres_df.sample(frac=1)
        # filter dataframe based on genre
        selected_movies = all_genres_df_2[all_genres_df_2['genres'].str.contains("Comedy")]
        for i in range(3):
            my_ratings, all_genres_df_2 = utils.get_ratings_from_user_2(movieList, i, selected_movies, my_ratings, all_genres_df_2)

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
        W, X, b = utils.train_data(Y, Ynorm, R)  # TRAIN
        st.subheader('These are the predictions for your own ratings.')
        my_predictions = utils.prediction(W, X, b, Ymean, my_ratings, movieList)
        st.subheader('Recommended movies! Enjoy!')
        utils.give_recommendation(my_predictions, my_rated, movieList)


    # button1 = st.button("Train Model!")
    # if st.session_state.get('button') != True:
    #     st.session_state['button'] = button1 # Saved the state
    # if st.session_state['button'] == True:
    #     st.write('Your data is added. Retraining to give you personal recommendations:) ')
    #     my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
    #     Y = np.c_[my_ratings, Y]  # Add new user ratings to Y
    #     R = np.c_[(my_ratings != 0).astype(int), R]  # Add new user indicator matrix to R
    #     Ynorm, Ymean = utils.normalizeRatings(Y, R)  # Normalize the Dataset
    #     W, X, b = utils.train_data(Y, Ynorm, R)
    #     if st.button("Show predictions and Recommendations!"):
    #         st.write('These are the predictions for your own ratings.')
    #         my_predictions = utils.prediction(W, X, b, Ymean, my_ratings, movieList)
    #         st.write('Recommended movies! Enjoy!')
    #         utils.give_recommendation(my_predictions, my_rated, movieList)

    # # How to make nested buttons in Streamlit with Session State: https://www.youtube.com/watch?v=XWKAt1QlRyI
    # # Initialize state
    # if "button_clicked" not in st.session_state:
    #     st.session_state.button_clicked = False
    #
    # def callback():
    #     # button was clicked
    #     st.session_state.button_clicked = True
    #
    # if (st.button("Train Model!", on_click=callback()) or st.session_state.button_clicked):
    #     st.write('Your data is added. Retraining to give you personal recommendations:) ')
    #     my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
    #     # Add new user ratings to Y
    #     Y = np.c_[my_ratings, Y]
    #     # Add new user indicator matrix to R
    #     R = np.c_[(my_ratings != 0).astype(int), R]
    #     # Normalize the Dataset
    #     Ynorm, Ymean = utils.normalizeRatings(Y, R)
    #     W, X, b = utils.train_data(Y, Ynorm, R)
    #
    #     if st.button("Show Prediction"):
    #         st.write('These are the predictions for your own ratings.')
    #         my_predictions = utils.prediction(W, X, b, Ymean, my_ratings, movieList)
    #
    #         if st.button('Show Recommendation'):
    #             st.write('Recommended movies! Enjoy!')
    #             utils.give_recommendation(my_predictions, my_rated, movieList)


    # train_button = st.button("Train Model!")
    # st.write('Your data is added. Retraining to give you personal recommendations:) ')
    # if train_button:
    #     my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
    #     # Add new user ratings to Y
    #     Y = np.c_[my_ratings, Y]
    #     # Add new user indicator matrix to R
    #     R = np.c_[(my_ratings != 0).astype(int), R]
    #     # Normalize the Dataset
    #     Ynorm, Ymean = utils.normalizeRatings(Y, R)
    #
    #     W, X, b = utils.train_data(Y, Ynorm, R)
    #
    # pred_button = st.button("Show Prediction")
    # if pred_button:
    #     st.write('These are the predictions for your own ratings.')
    #     my_predictions = utils.prediction(W, X, b, Ymean, my_ratings, movieList)
    #
    # recommendation_button = st.button('Show Recommendation')
    # if recommendation_button:
    #     st.write('Recommended movies! Enjoy!')
    #     utils.give_recommendation(my_predictions, my_rated, movieList)


# import streamlit as st
#
# if "page" not in st.session_state:
#     st.session_state.page = 0
#
# def nextpage(): st.session_state.page += 1
# def restart(): st.session_state.page = 0
#
# placeholder = st.empty()
# st.button("Next",on_click=nextpage,disabled=(st.session_state.page > 3))
#
# if st.session_state.page == 0:
#     # Replace the placeholder with some text:
#     placeholder.text(f"Hello, this is page {st.session_state.page}")
#
# elif st.session_state.page == 1:
#     # Replace the text with a chart:
#     placeholder.line_chart({"data": [1, 5, 2, 6]})
#
# elif st.session_state.page == 2:
# # Replace the chart with several elements:
#     with placeholder.container():
#         st.write("This is one element")
#         st.write("This is another")
#         st.metric("Page:", value=st.session_state.page)
#
# elif st.session_state.page == 3:
#     placeholder.markdown(r"$f(x) = \exp{\left(x^🐈\right)}$")
#
# else:
#     with placeholder:
#         st.write("This is the end")
#         st.button("Restart",on_click=restart)