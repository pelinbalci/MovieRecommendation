import streamlit as st
import utils
from matplotlib import pyplot as plt
import pandas as pd


def show_explore_page():
    st.write("""## Best Rated Movies""")
    df_ratings, df_ratings_mean, df_movie = utils.read_data()
    num_users, num_movies, movieList = utils.get_info(df_ratings, df_ratings_mean)
    best_ = df_ratings_mean[df_ratings_mean.number_of_ratings>=20]
    best_ = best_.sort_values(by='mean_rating', ascending=False)
    st.table(best_[['title', 'genres', 'mean_rating', 'number_of_ratings']].head(10))

    st.write("""## Most rated movie: """)
    most_rated = df_ratings_mean[df_ratings_mean.number_of_ratings == df_ratings_mean.number_of_ratings.max()]
    st.write(most_rated.title.item())
    idx = most_rated.movieId.item()
    most_rated_df = df_ratings[df_ratings.movieId == idx]
    most_rated_df['year'] = pd.DatetimeIndex(most_rated_df['datetime']).year
    agg_df = most_rated_df.groupby('year').agg({'rating': 'mean'})
    agg_df = agg_df.reset_index()
    fig1, ax1 = plt.subplots()
    ax1.plot(agg_df.year, agg_df.rating)
    st.pyplot(fig1)
