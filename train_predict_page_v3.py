import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

import utils


# --------------------------------------------------------------------------- #
# Data loading (cached: the dataset never changes while the app is running)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading MovieLens…")
def load_everything():
    df_ratings, df_ratings_mean, _ = utils.read_data()
    num_users, num_movies, movieList = utils.get_info(df_ratings, df_ratings_mean)
    Y, R, _ = utils.create_matrices(df_ratings, num_movies)
    all_genres_df, list_genre = utils.prepare_selected_movies(df_ratings_mean.copy())
    return Y, R, movieList, num_movies, all_genres_df, list_genre


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
.hero h1 {
    font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: 0 0 .25rem 0;
    background: linear-gradient(90deg, #f43f5e 0%, #f97316 50%, #eab308 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { opacity: .7; font-size: 1.05rem; margin: 0; }
.step { display:flex; align-items:center; gap:.6rem; margin: 1.6rem 0 .6rem 0; }
.step .n {
    width: 1.8rem; height: 1.8rem; border-radius: 999px; display:grid; place-items:center;
    background: linear-gradient(135deg,#f43f5e,#f97316); color:#fff; font-weight:700; font-size:.9rem;
}
.step h3 { margin:0; font-size:1.25rem; }
.movie-title { font-weight: 700; font-size: 1.02rem; line-height: 1.25; margin-bottom: .15rem; }
.movie-meta { opacity: .65; font-size: .85rem; margin-bottom: .45rem; }
.pills { display:flex; flex-wrap:wrap; gap:.3rem; margin-bottom:.5rem; }
.pill {
    font-size:.72rem; padding:.12rem .55rem; border-radius:999px;
    background: rgba(249,115,22,.14); color:#f97316; border:1px solid rgba(249,115,22,.35);
}
.rec-rank { font-size:1.6rem; font-weight:800; opacity:.35; line-height:1; }
</style>
"""


def step(n, title, caption=None):
    st.markdown(
        f'<div class="step"><div class="n">{n}</div><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def genre_pills(genres: str) -> str:
    return '<div class="pills">' + "".join(
        f'<span class="pill">{g}</span>' for g in str(genres).split("|")
    ) + "</div>"


def year_of(row) -> str:
    try:
        return str(pd.Timestamp(row.release_year).year)
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def show_train_predict_page_v3():
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="hero"><h1>Find your next movie</h1>'
        "<p>Rate a handful of films you know. A collaborative-filtering model "
        "trained live with TensorFlow does the rest.</p></div>",
        unsafe_allow_html=True,
    )

    # Hyper-parameters from the "Tune the Model" page
    iteration_number = st.session_state.get("iteration_number", 100)
    feature_number = st.session_state.get("feature_number", 100)
    selected_optimizer = st.session_state.get(
        "selected_optimizer", keras.optimizers.Adam(learning_rate=1e-1)
    )

    Y, R, movieList, num_movies, all_genres_df, list_genre = load_everything()
    all_genres_df_2 = all_genres_df.sort_values(by="number_of_ratings", ascending=False)

    # Persistent ratings: {movie_id_2: rating 1-5}
    ratings = st.session_state.setdefault("user_ratings", {})

    # ------------------------------------------------------------------ Step 1
    step(1, "Choose what to rate")

    c1, c2 = st.columns([3, 1])
    with c1:
        selected_genres = st.pills(
            "Genres", list_genre, selection_mode="multi",
            help="Leave empty to see the most-rated films of all genres",
        ) or []
    with c2:
        movienumber = st.slider("Movies shown", 6, 20, 9, step=3)
        randomstate = st.number_input(
            "Shuffle seed", 1, 100, 42,
            help="Change this to get a different set of films",
        )
    st.session_state["selected_genre"] = [str(g) for g in selected_genres]

    if not selected_genres:
        pool = all_genres_df_2.head(100)
    else:
        pool = utils.filter_genre(selected_genres, all_genres_df_2)
    pool = pool.sample(len(pool), random_state=int(randomstate))
    shown = pool.head(movienumber)

    # ------------------------------------------------------------------ Step 2
    step(2, "Rate the ones you've seen", "Skip anything you haven't watched.")

    cols = st.columns(3)
    for idx, (_, row) in enumerate(shown.iterrows()):
        mid = int(row.movie_id_2)
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f'<div class="movie-title">{row.title}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="movie-meta">{year_of(row)} · ★ {row.mean_rating:.1f} '
                    f'· {int(row.number_of_ratings):,} ratings</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(genre_pills(row.genres), unsafe_allow_html=True)
                prev = ratings.get(mid)
                key = f"rate_{mid}"
                if key not in st.session_state and prev:
                    st.session_state[key] = prev - 1
                stars = st.feedback("stars", key=key)
                if stars is None:
                    ratings.pop(mid, None)
                else:
                    ratings[mid] = stars + 1

    n_rated = len(ratings)
    m1, m2 = st.columns([4, 1])
    with m1:
        if n_rated == 0:
            st.info("Rate at least one film for a personalised result.")
        else:
            st.success(f"**{n_rated}** film{'s' if n_rated != 1 else ''} rated — ratings are kept while you browse.")
    with m2:
        if st.button("Clear ratings", width="stretch", disabled=n_rated == 0):
            for k in list(st.session_state):
                if k.startswith("rate_"):
                    del st.session_state[k]
            ratings.clear()
            st.rerun()

    # ------------------------------------------------------------------ Step 3
    step(3, "Get recommendations")

    go = st.button("✨ Recommend movies", type="primary", width="stretch")
    if not go:
        return

    my_ratings = np.zeros(num_movies)
    for mid, r in ratings.items():
        my_ratings[mid] = r
    my_rated = [i for i, r in enumerate(my_ratings) if r > 0]

    with st.status("Training the model…", expanded=True) as status:
        st.write("Adding your ratings to the matrix")
        Yx = np.c_[my_ratings, Y]
        Rx = np.c_[(my_ratings != 0).astype(int), R]
        st.write("Normalising")
        Ynorm, Ymean = utils.normalizeRatings(Yx, Rx)
        st.write(f"Running {iteration_number} iterations with {feature_number} features")
        W, X, b = utils.train_data(Yx, Ynorm, Rx, selected_optimizer, iteration_number, feature_number)
        st.write("Scoring every movie for you")
        pm = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy() + Ymean
        my_predictions = pm[:, 0]
        status.update(label="Done", state="complete", expanded=False)

    # Your ratings vs what the model now predicts for them
    if my_rated:
        with st.expander("How well did the model learn your taste?"):
            fit = pd.DataFrame({
                "Movie": [movieList[i] for i in my_rated],
                "Your rating": [my_ratings[i] for i in my_rated],
                "Model's guess": [my_predictions[i] for i in my_rated],
            })
            st.dataframe(
                fit, hide_index=True, width="stretch",
                column_config={
                    "Your rating": st.column_config.NumberColumn(format="%d ★"),
                    "Model's guess": st.column_config.ProgressColumn(
                        format="%.2f", min_value=0, max_value=5),
                },
            )

    # Build recommendation list
    order = np.argsort(-my_predictions)
    top = [(movieList[j], my_predictions[j]) for j in order[:100] if j not in my_rated]
    recs = pd.DataFrame(top, columns=["title", "prediction"])
    recs = recs.merge(all_genres_df_2, on="title", how="inner")

    personalised = bool(selected_genres) and n_rated > 0
    if selected_genres:
        filtered = utils.filter_genre(selected_genres, recs)
        if len(filtered) == 0:
            st.warning("Nothing matched your genres — showing the overall top picks instead.")
        else:
            recs = filtered.sort_values("prediction", ascending=False)
    recs = recs.head(10).reset_index(drop=True)

    st.markdown("### Your top 10")
    if not personalised:
        st.caption("Rate some films and pick a genre for a personalised list.")

    for i, row in recs.iterrows():
        with st.container(border=True):
            a, b_, c = st.columns([0.6, 6, 2])
            a.markdown(f'<div class="rec-rank">{i + 1}</div>', unsafe_allow_html=True)
            b_.markdown(f'<div class="movie-title">{row.title}</div>', unsafe_allow_html=True)
            b_.markdown(genre_pills(row.genres), unsafe_allow_html=True)
            c.metric("Predicted", f"{row.prediction:.2f} ★")