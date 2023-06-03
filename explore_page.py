import streamlit as st
import utils
from matplotlib import pyplot as plt
import pandas as pd
import altair as alt

def show_explore_page():
    st.write("""## General Stats""")
    df_ratings, df_ratings_mean, df_movie = utils.read_data()
    num_users, num_movies, movieList = utils.get_info(df_ratings, df_ratings_mean)
    df_summary = pd.DataFrame({'Total User': [num_users],
                               'Number of Movie': [num_movies],
                               'How many Ratings': [df_ratings_mean.number_of_ratings.sum()]})
    st.table(df_summary)

    st.write("""### Best Rated Movies""")
    best_ = df_ratings_mean[df_ratings_mean.number_of_ratings>=20]
    best_ = best_.sort_values(by='mean_rating', ascending=False)
    st.table(best_[['title', 'mean_rating', 'number_of_ratings']].head(5))


    # Prepera data
    most_rated = df_ratings_mean[df_ratings_mean.number_of_ratings == df_ratings_mean.number_of_ratings.max()]
    idx = most_rated.movieId.item()
    most_rated_df = df_ratings[df_ratings.movieId == idx]
    # get year
    most_rated_df['year'] = pd.to_datetime(most_rated_df["datetime"].dt.strftime('%Y'))
    # aggregate
    agg_df = most_rated_df.groupby('year').agg({'rating': 'mean'})
    agg_df = agg_df.reset_index()

    # Plot
    st.write(f"Most rated movie is: {most_rated.title.item()}")
    # fig1, ax1 = plt.subplots(figsize=(15,8))
    # ax1.plot(agg_df.year, agg_df.rating)
    # st.pyplot(fig1)

    line= alt.Chart(agg_df).mark_line().encode(
        x=alt.X('year:T', axis=alt.Axis(format="%Y")),
        y='rating',
    )
    st.altair_chart(line, use_container_width=True)

    st.write("Ratings for number of ratings >100")
    filtered_ratings = df_ratings_mean[df_ratings_mean.number_of_ratings > 100]
    # filtered_ratings = df_ratings_mean[df_ratings_mean.release_year > 1990]

    c = alt.Chart(filtered_ratings).mark_circle().encode(
        x=alt.X('release_year:T', axis=alt.Axis(format="%Y")),
        y=alt.Y('mean_rating'),
        size='number_of_ratings',
        tooltip=['title'])

    st.altair_chart(c, use_container_width=True)
