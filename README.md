# Personalized Movie Recommendation System

Link: https://pelinbalci-movierecommendation.streamlit.app/

A movie recommendation application built with **TensorFlow** and **Streamlit** using **Collaborative Filtering** with Matrix Factorization.

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
  - [Matrix Factorization](#matrix-factorization)
  - [Cost Function](#cost-function)
  - [Training Process](#training-process)
- [Project Structure](#project-structure)
- [Functions Reference](#functions-reference)
  - [Data Processing](#data-processing)
  - [Model Training](#model-training)
  - [Prediction & Recommendation](#prediction--recommendation)
  - [User Interface](#user-interface)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [References](#references)

---

## Overview

This application provides personalized movie recommendations by learning user preferences through collaborative filtering. Users rate a selection of movies, and the system trains a neural network model to predict ratings for unseen movies, generating tailored recommendations.

### Key Features

- Interactive movie rating interface
- Genre-based filtering
- Customizable hyperparameters (iterations, features, optimizer)
- Real-time model training with TensorFlow
- Data exploration and visualization

---

## Algorithm

### Matrix Factorization

The recommendation engine uses **Collaborative Filtering** via matrix factorization. The core idea is to decompose the user-movie rating matrix into two lower-dimensional matrices that capture latent features.

#### Mathematical Representation

Given:
- **Y** ∈ ℝ^(n_movies × n_users): Rating matrix
- **R** ∈ {0,1}^(n_movies × n_users): Binary indicator matrix (1 if rated, 0 otherwise)

We learn:
- **X** ∈ ℝ^(n_movies × n_features): Movie feature matrix
- **W** ∈ ℝ^(n_users × n_features): User preference matrix  
- **b** ∈ ℝ^(1 × n_users): User bias vector

#### Prediction Formula

The predicted rating for user *j* on movie *i* is:

```
ŷ(i,j) = X[i] · W[j]ᵀ + b[j]
```

Where:
- `X[i]` represents the learned feature vector for movie *i*
- `W[j]` represents the learned preference vector for user *j*
- `b[j]` is the bias term for user *j*

### Cost Function

The model minimizes a regularized mean squared error loss:

```
J = (1/2) Σᵢⱼ R(i,j) · (X[i]·W[j]ᵀ + b[j] - Y(i,j))² + (λ/2)(||X||² + ||W||²)
```

Where:
- The first term measures prediction error on known ratings
- The regularization term (λ) prevents overfitting by penalizing large parameter values

#### Vectorized Implementation

```python
j = (X @ W.T + b - Y) * R
J = 0.5 * tf.reduce_sum(j ** 2) + (λ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
```

### Training Process

1. **Normalization**: Subtract mean rating per movie to handle cold-start problem
2. **Initialization**: Random initialization of W, X, and b using TensorFlow Variables
3. **Optimization**: Gradient descent using TensorFlow's GradientTape for automatic differentiation
4. **Prediction**: Compute predictions and restore mean values

```
Training Flow:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│ Normalize Y │ -> │ Initialize   │ -> │ Gradient    │ -> │ Predict &  │
│   Matrix    │    │   W, X, b    │    │   Descent   │    │ Recommend  │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
```

---

## Project Structure

```
├── app.py                    # Main Streamlit application entry point
├── train_predict_page_v2.py  # Recommendation page (rating & prediction UI)
├── tuning_page.py            # Hyperparameter tuning interface
├── explore_page.py           # Data exploration and visualization
├── tutorial.py               # User guide and documentation
├── references_page.py        # Credits and references
├── utils.py                  # Core algorithms and utility functions
└── data/
    └── ml-latest-small/      # MovieLens dataset
        ├── movies.csv
        └── ratings.csv
```

---

## Functions Reference

### Data Processing

#### `read_data()`
Loads and preprocesses the MovieLens dataset.

| Returns | Type | Description |
|---------|------|-------------|
| df_ratings | DataFrame | User ratings with timestamps |
| df_ratings_mean | DataFrame | Aggregated movie statistics |
| df_movie | DataFrame | Movie metadata |

**Operations:**
- Loads movies.csv and ratings.csv
- Extracts release year from movie titles
- Calculates mean ratings and rating counts per movie
- Handles duplicate movies
- Creates new sequential movie IDs

---

#### `get_info(df_ratings, df_ratings_mean)`
Extracts basic dataset information.

| Parameter | Type | Description |
|-----------|------|-------------|
| df_ratings | DataFrame | Ratings dataframe |
| df_ratings_mean | DataFrame | Mean ratings dataframe |

| Returns | Type | Description |
|---------|------|-------------|
| num_users | int | Total number of users |
| num_movies | int | Total number of movies |
| movieList | list | List of movie titles |

---

#### `create_matrices(df_ratings, num_movies)`
Creates the rating and indicator matrices for training.

| Parameter | Type | Description |
|-----------|------|-------------|
| df_ratings | DataFrame | Ratings dataframe |
| num_movies | int | Number of movies |

| Returns | Type | Description |
|---------|------|-------------|
| Y | ndarray | Rating matrix (movies × users) |
| R | ndarray | Binary indicator matrix |
| my_ratings | ndarray | Zero-initialized array for new user ratings |

---

#### `normalizeRatings(Y, R)`
Normalizes the rating matrix by subtracting mean rating per movie.

| Parameter | Type | Description |
|-----------|------|-------------|
| Y | ndarray | Rating matrix |
| R | ndarray | Indicator matrix |

| Returns | Type | Description |
|---------|------|-------------|
| Ynorm | ndarray | Normalized rating matrix |
| Ymean | ndarray | Mean rating per movie |

**Purpose:** Ensures unrated movies default to average rating, addressing cold-start problem.

---

#### `prepare_selected_movies(df_ratings_mean)`
Prepares movies for user selection by genre.

| Parameter | Type | Description |
|-----------|------|-------------|
| df_ratings_mean | DataFrame | Mean ratings dataframe |

| Returns | Type | Description |
|---------|------|-------------|
| all_genres_df | DataFrame | Movies organized by genre |
| list_genre | list | Available genre types |

**Operations:**
- Extracts unique genres
- Creates one-hot encoded genre columns
- Sorts movies by popularity within each genre

---

#### `filter_genre(selected_genres, all_genres_df_2)`
Filters movies based on selected genres.

| Parameter | Type | Description |
|-----------|------|-------------|
| selected_genres | list | User-selected genres |
| all_genres_df_2 | DataFrame | All movies dataframe |

| Returns | Type | Description |
|---------|------|-------------|
| df | DataFrame | Filtered movies (top 50 per genre) |

---

### Model Training

#### `cofi_cost_func(X, W, b, Y, R, lambda_)`
Computes cost using explicit loops (for understanding).

| Parameter | Type | Description |
|-----------|------|-------------|
| X | ndarray | Movie feature matrix (n_movies × n_features) |
| W | ndarray | User preference matrix (n_users × n_features) |
| b | ndarray | User bias vector (1 × n_users) |
| Y | ndarray | Rating matrix |
| R | ndarray | Indicator matrix |
| lambda_ | float | Regularization parameter |

| Returns | Type | Description |
|---------|------|-------------|
| J | float | Total cost value |

---

#### `cofi_cost_func_v(X, W, b, Y, R, lambda_)`
Vectorized cost function optimized for TensorFlow.

**Same parameters as `cofi_cost_func`**

**Implementation:**
```python
j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
J = 0.5 * tf.reduce_sum(j ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
```

---

#### `train_data(Y, Ynorm, R, selected_optimizer, iteration_number, feature_number)`
Trains the collaborative filtering model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Y | ndarray | - | Original rating matrix |
| Ynorm | ndarray | - | Normalized rating matrix |
| R | ndarray | - | Indicator matrix |
| selected_optimizer | Optimizer | Adam | TensorFlow optimizer |
| iteration_number | int | 100 | Training iterations |
| feature_number | int | 100 | Latent feature dimensions |

| Returns | Type | Description |
|---------|------|-------------|
| W | tf.Variable | Trained user preference matrix |
| X | tf.Variable | Trained movie feature matrix |
| b | tf.Variable | Trained user bias vector |

**Training Loop:**
1. Initialize W, X, b with random values
2. For each iteration:
   - Compute cost using GradientTape
   - Calculate gradients
   - Update parameters via optimizer

---

### Prediction & Recommendation

#### `prediction(W, X, b, Ymean, my_ratings, movieList)`
Generates predictions for all movies.

| Parameter | Type | Description |
|-----------|------|-------------|
| W | tf.Variable | User preference matrix |
| X | tf.Variable | Movie feature matrix |
| b | tf.Variable | User bias vector |
| Ymean | ndarray | Mean ratings (for denormalization) |
| my_ratings | ndarray | Current user's ratings |
| movieList | list | Movie titles |

| Returns | Type | Description |
|---------|------|-------------|
| my_predictions | ndarray | Predicted ratings for new user |

**Calculation:**
```python
p = X @ W.T + b      # Raw predictions
pm = p + Ymean       # Restore mean
my_predictions = pm[:, 0]  # New user's predictions
```

---

#### `give_recommendation(my_predictions, my_rated, movieList, all_genres_df_2)`
Generates final movie recommendations.

| Parameter | Type | Description |
|-----------|------|-------------|
| my_predictions | ndarray | Predicted ratings |
| my_rated | list | Indices of already-rated movies |
| movieList | list | Movie titles |
| all_genres_df_2 | DataFrame | Movie metadata |

**Output:** Displays top 10 recommended movies filtered by selected genres.

---

### User Interface

#### `get_ratings_from_user_2(movieList, i, selected_movies, my_ratings, all_genres_df)`
Collects movie ratings from the user interface.

| Parameter | Type | Description |
|-----------|------|-------------|
| movieList | list | All movie titles |
| i | int | Current movie index |
| selected_movies | DataFrame | Movies to display |
| my_ratings | ndarray | User ratings array |
| all_genres_df | DataFrame | Available movies |

| Returns | Type | Description |
|---------|------|-------------|
| my_ratings | ndarray | Updated ratings array |
| all_genres_df | DataFrame | Updated movie list (rated movie removed) |

---

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd movie-recommendation
```

2. Install dependencies:
```bash
pip install streamlit tensorflow pandas numpy matplotlib altair
```

3. Download the MovieLens dataset and place it in `data/ml-latest-small/`

4. Run the application:
```bash
streamlit run app.py
```

---

## Usage

1. **Recommendation Page**: Rate movies and get personalized recommendations
   - Adjust the "Magic Number" to see different movie selections
   - Select genres to filter recommendations
   - Rate movies from 0-5 (0 = not seen)
   - Click "Recommend Movies!" to train and get suggestions

2. **Tune the Model**: Customize hyperparameters
   - Number of iterations (70-100)
   - Number of features (70-100)
   - Optimizer type (Adam, SGD, RMSprop)

3. **Explore**: View dataset statistics and visualizations

---

## Dataset

This project uses the [MovieLens "ml-latest-small"](https://grouplens.org/datasets/movielens/latest/) dataset.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

---

## References

- [DeepLearning.AI - Unsupervised Learning, Recommenders, Reinforcement Learning](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

---

# License

This project is licensed under the MIT License - see the LICENSE file for details.

This project is intended for educational purposes. Feel free to use, modify, and distribute as permitted under the MIT 
License.
