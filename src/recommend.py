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


def recommend_movies(input_ratings, top_n=5):
    """
    Given a user’s watched movies and ratings, predict ratings for all movies and return top recommendations.
    """
    input_pairs = [pair.split(":") for pair in input_ratings.split(",")]
    input_pairs = [(int(movie), float(rating)) for movie, rating in input_pairs]

    # Convert movie IDs to indexes
    movie_indexes = [movie_to_index[movie] for movie, _ in input_pairs if movie in movie_to_index]
    movie_ratings = torch.tensor([rating for _, rating in input_pairs], dtype=torch.float32)

    # Ensure there are valid movies
    if not movie_indexes:
        print("No valid movies found in input!")
        return []

    # Extract the learned movie embeddings from the model
    movie_embeddings = model.movie_embedding.weight[movie_indexes]

    # Aggregate the embeddings (e.g., weighted average by ratings)
    user_embedding = torch.sum(movie_embeddings * movie_ratings.view(-1, 1), dim=0) / torch.sum(movie_ratings)

    # Predict scores for all movies
    all_movie_ids = list(movie_to_index.values())
    all_movie_embeddings = model.movie_embedding.weight[all_movie_ids]

    scores = torch.matmul(all_movie_embeddings, user_embedding)

    # Get top N recommended movies
    topN_indices = torch.argsort(scores, descending=True)[:top_n]
    recommended_movies = [movies.iloc[idx]["title"] for idx in topN_indices.numpy()]

    return recommended_movies
