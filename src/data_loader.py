import pandas as pd

# Define dataset path
DATASET_PATH = "/Users/tylerwilliams/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1"

def load_data():
    """Loads and preprocesses the MovieLens dataset."""
    ratings = pd.read_csv(f"{DATASET_PATH}/rating.csv")
    movies = pd.read_csv(f"{DATASET_PATH}/movie.csv")

    # Create mappings for user and movie indices
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    ratings["user_index"] = ratings["userId"].map(user_to_index)
    ratings["movie_index"] = ratings["movieId"].map(movie_to_index)

    return ratings, movies, user_to_index, movie_to_index
