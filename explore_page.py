import streamlit as st
import utils
import pandas as pd
import altair as alt


def show_explore_page():
    st.title("Explore the Movie Database")
    st.markdown("Dive into the data behind your recommendations!")

    # Load data
    df_ratings, df_ratings_mean, df_movie = utils.read_data()
    num_users, num_movies, movieList = utils.get_info(df_ratings, df_ratings_mean)

    st.markdown("---")

    # ==================== KEY METRICS ====================
    st.header("🎯 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="👥 Total Users",
            value=f"{num_users:,}"
        )

    with col2:
        st.metric(
            label="🎬 Total Movies",
            value=f"{num_movies:,}"
        )

    with col3:
        total_ratings = df_ratings_mean.number_of_ratings.sum()
        st.metric(
            label="⭐ Total Ratings",
            value=f"{total_ratings:,}"
        )

    with col4:
        avg_rating = df_ratings_mean.mean_rating.mean()
        st.metric(
            label="📈 Avg Rating",
            value=f"{avg_rating:.2f}"
        )

    st.markdown("---")

    # ==================== TOP MOVIES ====================
    st.header("🏆 Top Rated Movies")
    st.caption("Movies with at least 50 ratings")

    best_ = df_ratings_mean[df_ratings_mean.number_of_ratings >= 50].copy()
    best_ = best_.sort_values(by='mean_rating', ascending=False).head(10)

    if best_.empty:
        st.info("No movies meet the criteria (at least 50 ratings). Try lowering the threshold.")
    else:
        best_chart_data = best_[['title', 'mean_rating', 'number_of_ratings']].copy()
        best_chart_data['mean_rating'] = best_chart_data['mean_rating'].astype(float)
        best_chart_data['number_of_ratings'] = best_chart_data['number_of_ratings'].astype(int)

        # Optional: let Altair infer the domain
        top_movies_chart = alt.Chart(best_chart_data).mark_bar(
            cornerRadiusTopRight=10,
            cornerRadiusBottomRight=10
        ).encode(
            x=alt.X('mean_rating:Q', title='Average Rating'),  # removed fixed domain
            y=alt.Y('title:N', sort='-x', title=''),
            color=alt.Color('mean_rating:Q',
                            scale=alt.Scale(scheme='goldorange'),
                            legend=None),
            tooltip=[
                alt.Tooltip('title:N', title='Movie'),
                alt.Tooltip('mean_rating:Q', title='Rating', format='.2f'),
                alt.Tooltip('number_of_ratings:Q', title='# of Ratings')
            ]
        ).properties(
            height=400
        )

        st.altair_chart(top_movies_chart, use_container_width=True)

    st.markdown("---")

    # ==================== GENRE ANALYSIS ====================
    st.header("🎭 Genre Distribution")

    all_genres_df, list_genre = utils.prepare_selected_movies(df_ratings_mean)
    genre_movies = all_genres_df[list_genre].sum().reset_index()
    genre_movies = genre_movies[genre_movies['index'].isin(list_genre)]
    genre_movies.columns = ['Genre', 'Count']
    genre_movies = genre_movies.sort_values('Count', ascending=False)

    # Horizontal bar chart for genres (full width)
    genre_chart = alt.Chart(genre_movies).mark_bar(
        cornerRadiusTopRight=8,
        cornerRadiusBottomRight=8
    ).encode(
        x=alt.X('Count:Q', title='Number of Movies'),
        y=alt.Y('Genre:N', sort='-x', title=''),
        color=alt.Color('Count:Q',
                        scale=alt.Scale(scheme='viridis'),
                        legend=None),
        tooltip=[
            alt.Tooltip('Genre:N', title='Genre'),
            alt.Tooltip('Count:Q', title='Movies')
        ]
    ).properties(
        height=500
    )

    st.altair_chart(genre_chart, use_container_width=True)

    st.markdown("---")

    # ==================== MOST RATED MOVIE ANALYSIS ====================
    st.header("🌟 Most Popular Movie Analysis")

    most_rated = df_ratings_mean[df_ratings_mean.number_of_ratings == df_ratings_mean.number_of_ratings.max()]
    movie_title = most_rated.title.item()
    movie_ratings_count = most_rated.number_of_ratings.item()
    movie_avg_rating = most_rated.mean_rating.item()

    # Movie info card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🎬 **{movie_title}**")
    with col2:
        st.metric("Total Ratings", f"{movie_ratings_count:,}")
    with col3:
        st.metric("Average Rating", f"{movie_avg_rating:.2f} ⭐")

    # Rating trend over time
    idx = most_rated.movieId.item()
    most_rated_df = df_ratings[df_ratings.movieId == idx].copy()
    most_rated_df['year'] = pd.to_datetime(most_rated_df["datetime"].dt.strftime('%Y'))
    agg_df = most_rated_df.groupby('year').agg({'rating': 'mean'}).reset_index()

    st.markdown(f"**Rating Trend Over Time**")

    line_chart = alt.Chart(agg_df).mark_line(
        point=alt.OverlayMarkDef(size=60, filled=True),
        strokeWidth=3,
        color='#FF6B6B'
    ).encode(
        x=alt.X('year:T',
                axis=alt.Axis(format="%Y", title='Year')),
        y=alt.Y('rating:Q',
                scale=alt.Scale(domain=[3, 5]),
                title='Average Rating'),
        tooltip=[
            alt.Tooltip('year:T', title='Year', format='%Y'),
            alt.Tooltip('rating:Q', title='Avg Rating', format='.2f')
        ]
    ).properties(
        height=300
    )

    st.altair_chart(line_chart, use_container_width=True)

    st.markdown("---")

    # ==================== INTERACTIVE SCATTER PLOT ====================
    st.header("🔍 Explore Movies")
    st.markdown("Discover movies by rating and popularity")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_ratings = st.slider(
            "Minimum number of ratings",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )
    with col2:
        rating_range = st.slider(
            "Rating range",
            min_value=1.0,
            max_value=5.0,
            value=(3.0, 5.0),
            step=0.5
        )

    # Filter data
    filtered_ratings = df_ratings_mean[
        (df_ratings_mean.number_of_ratings >= min_ratings) &
        (df_ratings_mean.mean_rating >= rating_range[0]) &
        (df_ratings_mean.mean_rating <= rating_range[1])
        ]

    st.caption(f"Showing {len(filtered_ratings)} movies")

    # Interactive scatter plot
    scatter = alt.Chart(filtered_ratings).mark_circle().encode(
        x=alt.X('release_year:T',
                axis=alt.Axis(format="%Y", title='Release Year')),
        y=alt.Y('mean_rating:Q',
                title='Average Rating',
                scale=alt.Scale(domain=[rating_range[0] - 0.5, 5.2])),
        size=alt.Size('number_of_ratings:Q',
                      scale=alt.Scale(range=[50, 500]),
                      legend=alt.Legend(title="# Ratings")),
        color=alt.Color('mean_rating:Q',
                        scale=alt.Scale(scheme='redyellowgreen', domain=[1, 5]),
                        legend=alt.Legend(title="Rating")),
        tooltip=[
            alt.Tooltip('title:N', title='Movie'),
            alt.Tooltip('release_year:T', title='Year', format='%Y'),
            alt.Tooltip('mean_rating:Q', title='Rating', format='.2f'),
            alt.Tooltip('number_of_ratings:Q', title='# Ratings'),
            alt.Tooltip('genres:N', title='Genres')
        ]
    ).properties(
        height=500
    ).interactive()

    st.altair_chart(scatter, use_container_width=True)

    st.markdown("---")

    # ==================== RATING DISTRIBUTION ====================
    st.header("📊 Rating Distribution")

    # Create histogram of ratings
    hist_data = df_ratings_mean.copy()
    hist_data['rating_bin'] = pd.cut(
        hist_data['mean_rating'],
        bins=[0, 1, 2, 3, 4, 5],
        labels=['1⭐', '2⭐', '3⭐', '4⭐', '5⭐']
    )
    rating_counts = hist_data['rating_bin'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['Rating', 'Count']

    histogram = alt.Chart(rating_counts).mark_bar(
        cornerRadiusTopLeft=10,
        cornerRadiusTopRight=10,
        color='#4ECDC4'
    ).encode(
        x=alt.X('Rating:N', title='Rating Range'),
        y=alt.Y('Count:Q', title='Number of Movies'),
        tooltip=[
            alt.Tooltip('Rating:N', title='Rating'),
            alt.Tooltip('Count:Q', title='Movies')
        ]
    ).properties(
        height=300
    )

    st.altair_chart(histogram, use_container_width=True)

    # ==================== FOOTER ====================
    st.markdown("---")
    st.caption("💡 Tip: Hover over charts for more details! Use the interactive scatter plot to discover hidden gems.")