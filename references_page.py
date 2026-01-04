import streamlit as st


def show_references():
    st.title("📚 References")

    st.divider()

    # Acknowledgments Section
    st.header("🙏 Acknowledgments")

    st.markdown("""
    This project was built as part of my learning journey in machine learning and recommendation systems.

    **Course & Instructor:**
    - [Unsupervised Learning, Recommenders, Reinforcement Learning](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning) 
      by **DeepLearning.AI** on Coursera
    - Special thanks to **Andrew Ng** for the excellent lecture materials

    **What I Learned vs Built:**
    | From Course | My Contribution |
    |-------------|-----------------|
    | Training module | Data preparation pipeline |
    | Vectorized cost function | User input interface |
    | Algorithm theory | Modular code structure |
    | - | Streamlit UI & deployment |
    | - | Genre filtering system |
    """)

    st.divider()

    # Dataset Section
    st.header("🎬 Dataset")

    st.markdown("""
    This project uses the [MovieLens "ml-latest-small"](https://grouplens.org/datasets/movielens/latest/) dataset.

    > F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* 
    > ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. 
    > [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)
    """)

    st.divider()

    # Resources Section
    st.header("🔗 Helpful Resources")

    st.markdown("""
    - [Streamlit Documentation](https://docs.streamlit.io/) — UI framework used for this app
    - [TensorFlow Documentation](https://www.tensorflow.org/api_docs) — ML framework for training
    - [Patrick Loeber's ML App Tutorial](https://github.com/patrickloeber/ml-app-salaryprediction) — Streamlit deployment inspiration
    - [Streamlit Discussion Forum](https://discuss.streamlit.io/) — Community support
    """)

    st.divider()

    # Footer
    st.markdown("""
    ---
    Made with ❤️ using Streamlit and TensorFlow

    *Found a bug or have a suggestion? Feel free to open an issue on GitHub!*
    """)