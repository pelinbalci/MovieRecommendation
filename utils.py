import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import streamlit as st

def read_data():
    " Read & Create Data"
    # Read data
    df_movie = pd.read_csv('./data/ml-latest-small/movies.csv')
    df_ratings = pd.read_csv('./data/ml-latest-small/ratings.csv')
    print(df_movie.shape)
    print(df_ratings.shape)

    df_ratings['datetime'] = pd.to_datetime(df_ratings['timestamp'], unit='s')

    # Create mean ratings
    df_temp = df_ratings.groupby('movieId').agg({'movieId': 'count', 'rating': 'mean'})
    df_temp.rename(columns={'movieId': 'number_of_ratings', 'rating': 'mean_rating'}, inplace=True)
    df_ratings_mean = df_temp.reset_index()

    # Merge
    df_ratings_mean = pd.merge(df_ratings_mean, df_movie, on='movieId', how='left')

    # flag the last duplicated movie based on title
    df_ratings_mean['Last_dup1'] = np.where(df_ratings_mean['title'].duplicated(keep='first'), 1, 0)
    # movieIds of the last duplicated movies
    duplicated_movies_list = list(df_ratings_mean[df_ratings_mean.Last_dup1 == 1]["movieId"])

    # remove last duplicated movies based on title
    df_ratings_mean = df_ratings_mean.drop_duplicates(subset=["title"], keep="first")

    # remove duplicated movies based on movieIds
    df_ratings = df_ratings[~df_ratings.movieId.isin(duplicated_movies_list)]

    # create new id
    df_ratings_mean = df_ratings_mean.reset_index()
    df_ratings_mean['movie_id_2'] = df_ratings_mean.index

    print(len(list(set(df_ratings_mean.title))))
    print(len(list(set(df_ratings_mean.movieId))))
    print(len(list(set(df_ratings_mean.movie_id_2))))
    print(len(list(set(df_ratings.movieId))))

    return df_ratings, df_ratings_mean, df_movie


def get_info(df_ratings, df_ratings_mean):
    "Get general info"
    movieList = list(df_ratings_mean.title)
    num_movies = len(list(df_ratings.movieId.unique()))
    num_users = len(list(df_ratings.userId.unique()))
    return num_users, num_movies, movieList


def create_matrices(df_ratings, num_movies):
    "Create Matrices"

    y_matrix = df_ratings.pivot(index='movieId', columns='userId', values='rating')
    y_matrix =y_matrix.fillna(0)
    y_matrix = y_matrix.reset_index()
    y_matrix = y_matrix.drop('movieId', axis=1)

    df_ratings['temp'] = 1
    r_matrix = df_ratings.pivot(index='movieId', columns='userId', values='temp')
    r_matrix =r_matrix.fillna(0)
    r_matrix = r_matrix.reset_index()
    r_matrix = r_matrix.drop('movieId', axis=1)

    Y = y_matrix.to_numpy()
    R = r_matrix.to_numpy()

    print("matrix shape:", Y.shape)

    # Create series for input from user
    my_ratings = np.zeros(num_movies)
    return Y, R, my_ratings


# Cost function with loop
def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    for j in range(nu):
        for i in range(nm):
            pred = np.dot(W[j, :], X[i, :]) + b[0, j]
            squared_error = np.dot(R[i, j], np.square(pred - Y[i, j]))
            J += squared_error
    J /= 2
    reg_w = lambda_ * np.sum(np.square(W)) * 0.5
    reg_x = lambda_ * np.sum(np.square(X)) * 0.5

    J = J + reg_w + reg_x
    return J


# Vectorized cost function
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(j ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
    return J


def prepare_selected_movies(df_ratings_mean):
    """
    Return selected movies(Most rated 30 movies for each genre) to display on the screen.
    :param df_ratings_mean: dataframe includes mean ratings for each movie
    :return: selected movies
    """
    # Get unique genre types from df_movie
    genre_types = df_ratings_mean.genres
    genre_types_m = genre_types.str.split('|', expand=True)
    all_genres = []
    for i in range(10):
        all_genres.append(list(genre_types_m[i].unique()))

    # https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    list_genre = [item for sublist in all_genres for item in sublist]
    list_genre = list(set(list_genre))
    list_genre.remove("(no genres listed)")
    list_genre.remove(None)

    # Create columns for each genre type (1 - 0 encoded values)
    for col in list_genre:
        df_ratings_mean[col] = 0

    for col in list_genre:
        df_ratings_mean[col] = np.where(df_ratings_mean['genres'].str.contains(col) == True, 1, 0)

    # Selected movies for all genre types (Most rated 30 movies for each genre)
    all_genres = []
    for genre in list_genre:
        genre_temp = df_ratings_mean[np.array(df_ratings_mean.filter(regex=genre) == 1).reshape
        (len(df_ratings_mean), )].sort_values(
            by='number_of_ratings', ascending=False).head(30)

        # remove already selected movies
        genre_temp_movie_id = list(genre_temp["movie_id_2"])
        df_ratings_mean = df_ratings_mean[~df_ratings_mean.movie_id_2.isin(genre_temp_movie_id)]

        all_genres.append(genre_temp)
    all_genres_df = pd.concat(all_genres)

    return all_genres_df


# Inputs from user
def get_ratings_from_user(df_ratings_mean, genre, my_ratings):
    """
    This function select the movies based on the genre. Get top 5 movies and then remove them from the dataset.
    Since, a movie may has more than one genres.

    df_ratings_mean: (df) the rating matrix
    genre: (str) specific genre
    """
    selected_movies = df_ratings_mean[df_ratings_mean['genres'].str.contains(genre)].sort_values(
        by='number_of_ratings', ascending=False)
    selected_movies = selected_movies.head(30)
    # selected_movies = selected_movies.sample(frac=1)
    for i in range(5):
        print('NEW RATING')
        print('Movie:', selected_movies.title.iloc[i])
        print('Movie_id_2: {}, movieId: {}'.format(selected_movies.movie_id_2.iloc[0], selected_movies.movieId.iloc[0]))
        rating_i = st.number_input(selected_movies.title.iloc[i], min_value=0, max_value=5, step=1)
        current_movieId = selected_movies.movieId.iloc[i]
        current_movie_id_2 = selected_movies.movie_id_2.iloc[i]
        my_ratings[current_movie_id_2] = rating_i

        # remove movie after rating
        df_ratings_mean = df_ratings_mean[(df_ratings_mean.movieId != current_movieId)]
    return my_ratings, df_ratings_mean


def get_ratings_from_user_2(movieList, i, selected_movies, my_ratings, all_genres_df):
    print('NEW RATING')
    print('Movie:', selected_movies.title.iloc[i])
    print('Movie_id_2: {}, movieId: {}'.format(selected_movies.movie_id_2.iloc[0],
                                               selected_movies.movieId.iloc[0]))
    # get rating from user
    rating_i = st.number_input(selected_movies.title.iloc[i], min_value=0, max_value=5, step=1)

    # original movie id
    current_movieId = selected_movies.movieId.iloc[i]
    st.write("movieID", current_movieId)

    # store ratings based on movie_id_2
    current_movie_id_2 = selected_movies.movie_id_2.iloc[i]
    st.write("movie_id_'", current_movie_id_2)
    my_ratings[current_movie_id_2] = rating_i

    print('control')
    print(selected_movies[selected_movies.movie_id_2 == current_movie_id_2]['title'])
    print(movieList[current_movie_id_2])

    # remove movie not to show the same movie to user.
    all_genres_df = all_genres_df[(all_genres_df.movieId != current_movieId)]

    return my_ratings, all_genres_df


# Normalize Y matirx
def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)
    Ynorm = Y - np.multiply(Ymean, R)
    return (Ynorm, Ymean)


def train_data(Y, Ynorm, R):
    """

    :param Y: Y matrix includes original ratings and new user's ratings
    :param Ynorm:  Normalized Y matrix
    :param R: boolean matrix pf Y
    :return: trained parameters
    """
    #  Useful Values
    num_movies, num_users = Y.shape
    num_features = 5

    # Set Initial Parameters (W, X), use tf.Variable to track these variables
    tf.random.set_seed(1234)  # for consistent results
    W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
    b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=1e-1)

    iterations = 10
    lambda_ = 1
    for iter in range(iterations):
        # Use TensorFlow’s GradientTape
        # to record the operations used to compute the cost
        with tf.GradientTape() as tape:

            # Compute the cost (forward pass included in cost)
            cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss
        grads = tape.gradient(cost_value, [X, W, b])

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, [X, W, b]))

        # Log periodically.
        if iter % 20 == 0:
            st.write(f"Training loss at iteration {iter}: {cost_value:0.1f}")

    return W, X, b


def prediction(W, X, b, Ymean, my_ratings, movieList):
    # Make a prediction using trained weights and biases
    p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

    # restore the mean
    pm = p + Ymean

    # Find the predictions for the new user (user id is 0)
    my_predictions = pm[:, 0]

    st.write('\n\nOriginal ratings:\n')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            st.write(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')

    return my_predictions


def give_recommendation(my_predictions, my_rated, movieList):
    # sort predictions
    idx_sorted_pred = tf.argsort(my_predictions, direction='DESCENDING')

    for i in range(10):
        j = idx_sorted_pred[i]
        if j not in my_rated:
            st.write(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')
