import pandas as pd

# Define dataset path
DATASET_PATH = "/Users/tylerwilliams/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1"

def get_all_genres():
    """Returns a set of all unique genres in the dataset."""
    movies = pd.read_csv(f"{DATASET_PATH}/movie.csv")
    # MovieLens stores genres as '|' separated strings
    all_genres = set()
    for genres in movies['genres'].str.split('|'):
        all_genres.update(genres)
    return all_genres

def is_valid_genre(genre):
    """Checks if a genre exists in the dataset."""
    if not genre:
        return False
    all_genres = get_all_genres()
    return genre in all_genres

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
