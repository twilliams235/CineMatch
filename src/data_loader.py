import pandas as pd
import numpy as np

# Define dataset path
DATASET_PATH = "/Users/tylerwilliams/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1"

def load_data(threshold: float = 3.0):
    """
    Loads and preprocesses the MovieLens dataset.
    Adds an implicit label: 1 if rating > threshold else 0.
    """
    ratings = pd.read_csv(f"{DATASET_PATH}/rating.csv")
    movies = pd.read_csv(f"{DATASET_PATH}/movie.csv")

    # Create mappings for user and movie indices
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    ratings["user_index"] = ratings["userId"].map(user_to_index)
    ratings["movie_index"] = ratings["movieId"].map(movie_to_index)

    # implicit label for NCF BCE (positives: rating > threshold)
    ratings["implicit"] = (ratings["rating"] > threshold).astype(np.int32)

    return ratings, movies, user_to_index, movie_to_index
