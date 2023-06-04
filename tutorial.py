import streamlit as st


def show_tutorial():
    st.title("Quick Start")

    st.write("""
    You can select the pages in the sidebar on the left. 
    
    These are:
    
    - Recommendation_v2
    - Recommendation_v1
    - Recommendation_v0
    - Tune the Model
    - Explore
    - References
    """)

    st.write(""" You can directly use Recommendation_v2 and click the 'Recommend movies!' button to see the result. """)

    st.write("""The magic number you select creates randomness in the movie selection (It is basically randomstate. 
    If this remains the same, you may see the same recommendations and same movie ratings. Then, choose the number of movies you would like to 
     select by using the slider. 
     You can't see the movies unless you check the box: 'Show me movies' """)

    st.write("""Rate the movies from 1 to 5. If you haven't seen it, remain the rating as 0.""")

    st.write(""" You will see two tables after the model is trained. The first one shows the difference between original 
    ratings and the predictions. If there is
    a huge gap, you may try to increase the iteration number in the "Tune the Model" page. The second table shows the 
    recommended movies. Hopefully you enjoy the recommendations!!! =)""")

    st.title("Parameter Tuning")
    st.write("""
       Select the "Tune the Model" page on the left. The model is using TensorFlow. Wee need to specify the number of 
       iterations, the number of features of movie and optimization type. Their default values are 100, 100, and Adam 
       respectively. You can change them and check the difference between the recommendations. The easiest way to 
       understand the difference between parameters is to 
       check the original ratings vs predicted ratings. This table  will be shown after the model is trained.""")

    st.title('Notes')
    st.write("This website is not established for commercial purposes. Note that the dataset is not up to date. "
             "You may not find the newly released movies in it. ")

    st.title("Releases")

    st.write("""
    - v0: The first version doesn't have the randomness in the movie selection.
    - v1: Includes randomness. It also shows the samples from db and selected parameters. This page can be used for 
    development. 
    - v2: The parameters and some of the technical details are removed. The simplest version so far.  
    """)
