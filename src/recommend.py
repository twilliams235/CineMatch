import torch
import os
import pandas as pd
from src.data_loader import load_data
from src.model import RecommenderNN

# Load Data
ratings, movies, user_to_index, movie_to_index = load_data()

# Load Trained Model
num_users = len(user_to_index)
num_movies = len(movie_to_index)
model = RecommenderNN(num_users, num_movies)

# Check if the trained model exists before loading
model_path = "models/trained_model.pth"

if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()


def recommend_movies(user_movie_ids, user_ratings, top_n=5):
    """
    Given a user’s watched movies and ratings, predict ratings for all movies and return top recommendations.
    """
    user_idx = max(user_to_index.values()) + 1

    all_movies = pd.DataFrame({"movieId": list(movie_to_index.keys()), "movie_index": list(movie_to_index.values())})
    movie_indices = torch.tensor(all_movies["movie_index"].values, dtype=torch.long)
    user_tensor = torch.full((len(movie_indices),), user_idx, dtype=torch.long)

    with torch.no_grad():
        predicted_ratings = model(user_tensor, movie_indices).squeeze().numpy()

    all_movies["predicted_rating"] = predicted_ratings
    unwatched_movies = all_movies[~all_movies["movieId"].isin(user_movie_ids)]
    top_movies = unwatched_movies.sort_values(by="predicted_rating", ascending=False).head(top_n)
    top_movies = top_movies.merge(movies, on="movieId")[["movieId", "title", "predicted_rating"]]

    return top_movies
