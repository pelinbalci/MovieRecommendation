import streamlit as st
import numpy as np
import utils
from tensorflow import keras


def show_train_predict_page_v2():
    st.title("Personalized Movie Recommendation")
    st.subheader("with TensorFlow and Collaborative Filtering")

    st.markdown("---")

    # Tip box
    st.info("💡 **New here?** Visit the **Tutorial** page to learn how this works.")

    st.markdown("---")

    # Get session state values
    iteration_number = st.session_state.get("iteration_number", 100)
    feature_number = st.session_state.get("feature_number", 100)
    selected_optimizer = st.session_state.get("selected_optimizer", keras.optimizers.Adam(learning_rate=1e-1))

    # Load data
    df_ratings, df_ratings_mean, df_movie = utils.read_data()
    num_users, num_movies, movieList = utils.get_info(df_ratings, df_ratings_mean)
    Y, R, my_ratings = utils.create_matrices(df_ratings, num_movies)

    # Prepare dataset
    df_ratings_mean_temp = df_ratings_mean.copy()
    all_genres_df, list_genre = utils.prepare_selected_movies(df_ratings_mean_temp)

    # ==================== STEP 1 ====================
    st.subheader("Step 1: Customize Your Selection")

    randomstate_user = st.slider(
        'Magic Number',
        min_value=1,
        max_value=100,
        value=42,
        step=1,
        help="Change this to see different movies"
    )
    st.caption("Change this number to see different movies")

    movienumber_user = st.slider(
        'Number of Movies to Rate',
        min_value=6,
        max_value=20,
        value=6,
        step=1
    )

    st.session_state["randomstate"] = randomstate_user
    st.session_state["movienumber"] = movienumber_user
    randomstate = st.session_state.get("randomstate", 42)
    movienumber = st.session_state.get("movienumber", 6)

    # Genre selection
    selected_genres_user = st.multiselect(
        'Filter by Genre',
        list_genre,
        help="Select one or more genres to filter movies"
    )
    selected_genres = [str(genre) for genre in selected_genres_user]
    st.session_state['selected_genre'] = selected_genres

    # Get highest number of rated movies
    all_genres_df_2 = all_genres_df.sort_values(by="number_of_ratings", ascending=False)

    if not selected_genres:
        all_genres_df_3 = all_genres_df_2.copy().head(100)
        all_genres_df_3 = all_genres_df_3.sample(len(all_genres_df_3), random_state=randomstate)
    else:
        all_genres_df_3 = utils.filter_genre(selected_genres, all_genres_df_2)
        all_genres_df_3 = all_genres_df_3.sample(len(all_genres_df_3), random_state=randomstate)

    st.markdown("---")

    # ==================== STEP 2 ====================
    st.subheader("Step 2: Rate Movies")

    checkbox_b = st.checkbox("Click to reveal movies and rate them")
    st.caption("Give 0 if you haven't seen the movie")

    if checkbox_b:
        selected_movies = all_genres_df_3.head(movienumber)
        for i in range(movienumber):
            my_ratings, all_genres_df_3 = utils.get_ratings_from_user_2(
                movieList, i, selected_movies, my_ratings, all_genres_df_3
            )
        st.markdown("*If you don't like these movies, change the Magic Number!*")

    st.markdown("---")

    # ==================== STEP 3 ====================
    st.subheader("Step 3: Get Your Recommendations")

    train_button = st.button("RECOMMEND MOVIES!")

    if train_button:
        st.markdown("---")

        # Animated loading state
        st.subheader("🎬 Finding Your Perfect Movies...")

        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Preparing data
        status_text.text("📊 Preparing your ratings...")
        my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
        Y = np.c_[my_ratings, Y]
        R = np.c_[(my_ratings != 0).astype(int), R]
        progress_bar.progress(20)

        # Step 2: Normalizing
        status_text.text("⚖️ Normalizing data...")
        Ynorm, Ymean = utils.normalizeRatings(Y, R)
        progress_bar.progress(40)

        # Step 3: Training model
        status_text.text("🧠 Training the model with your preferences...")
        progress_bar.progress(60)
        W, X, b = utils.train_data(Y, Ynorm, R, selected_optimizer, iteration_number, feature_number)
        progress_bar.progress(80)

        # Step 4: Generating predictions
        status_text.text("🎯 Generating personalized recommendations...")
        my_predictions = utils.prediction(W, X, b, Ymean, my_ratings, movieList)
        progress_bar.progress(100)

        # Clear progress indicators
        status_text.text("✅ Complete!")

        # Show results
        st.markdown("---")
        st.subheader("🎉 Your Recommendations")
        st.write("Based on your ratings, we think you'll enjoy these movies:")
        utils.give_recommendation(my_predictions, my_rated, movieList, all_genres_df_2)

        # Success message
        st.success("Recommendations generated successfully!")