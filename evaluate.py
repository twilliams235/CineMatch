import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.model import RecommenderNN_BPR

def evaluate_user_preference(k=10):
    """Evaluates the model by splitting user test data and ranking remaining movies."""
    # Load Data
    ratings, movies, user_to_index, movie_to_index = load_data()

    # Load Trained Model
    model = RecommenderNN_BPR(num_users=len(user_to_index), num_movies=len(movie_to_index))
    model.load_state_dict(torch.load("models/trained_model_bpr.pth"))
    model.eval()

    # Prepare test data
    _, test_data = train_test_split(ratings, test_size=0.3, random_state=42)
    test_users = test_data["user_index"].unique()
    correct_predictions = 0
    total_users = 0

    with torch.no_grad():
        for user in test_users:
            user_data = test_data[test_data["user_index"] == user]
            if len(user_data) < 30:
                continue  # Skip users with too few ratings
            if len(user_data) > 100:
                continue  # Skip users with too few ratings

            # Split user's data in half into input movies and evaluation movies
            user_movies = user_data.sample(frac=1, random_state=42)  # Shuffle
            input_movies = user_movies.iloc[:len(user_movies) // 2]
            eval_movies = user_movies.iloc[len(user_movies) // 2:]

            # Convert movie IDs to indexes
            input_movie_indexes = torch.tensor(input_movies["movie_index"].values, dtype=torch.long)
            input_ratings = torch.tensor(input_movies["rating"].values, dtype=torch.float32)
            eval_movie_indexes = torch.tensor(eval_movies["movie_index"].values, dtype=torch.long)
            eval_ratings = torch.tensor(eval_movies["rating"].values, dtype=torch.float32)

            # Compute user preference vector
            movie_embeddings = model.movie_embedding(input_movie_indexes)
            user_embedding = torch.sum(movie_embeddings * input_ratings.view(-1, 1), dim=0) / torch.sum(input_ratings)

            # Compute similarity scores for evaluation movies
            eval_movie_embeddings = model.movie_embedding(eval_movie_indexes)
            scores = torch.matmul(eval_movie_embeddings, user_embedding)

            # Rank movies by predicted scores
            top_k_predicted = torch.argsort(scores, descending=True)[:k]
            top_k_actual = torch.argsort(eval_ratings, descending=True)[:k]

            correct_predictions += len(set(top_k_predicted.tolist()) & set(top_k_actual.tolist()))
            total_users += len(eval_movies)

    top_k_accuracy = correct_predictions / total_users if total_users > 0 else 0
    print(f"Top-{k} Accuracy: {top_k_accuracy:.4f}")

if __name__ == "__main__":
    evaluate_user_preference(k=10)