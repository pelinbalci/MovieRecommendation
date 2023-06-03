import streamlit as st
from tensorflow import keras


def show_tuning_page():
    st.title("Hyper-parameter Tuning")
    st.write("""### Let's start to choose hyper-parameters!""")
    st.write("The default values are used if you don't change anything:) ")

    iteration_number = st.number_input("Number of Iterations (default: 100)", min_value=10, max_value=100, step=10)
    feature_number = st.number_input("Number of Movie Features (default: 100)", min_value=10, max_value=100, step=10)
    opt_select = st.radio("Optimization Type (default: Adam)", ("Adam", "SGD", "Less Known"))

    if opt_select == "Adam":
        selected_optimizer = keras.optimizers.Adam(learning_rate=1e-1)
    elif opt_select == 'SGD':
        selected_optimizer = keras.optimizers.SGD(learning_rate=1e-1)
    else:
        selected_optimizer = keras.optimizers.RMSprop(learning_rate=1e-1)

    if st.button("Save"):
        st.session_state["iteration_number"] = iteration_number
        st.session_state["feature_number"] = feature_number
        st.session_state["selected_optimizer"] = selected_optimizer
        st.write('Your parameters are saved. Go to the Recommendation Page.')