import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.recommend import recommend_movies
from src.model import RecommenderNN
import os

def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hit_count = sum(1 for movie in recommended_k if movie in relevant_set)
    return hit_count / k

def evaluate_precision_at_k(k=5):
    """Evaluate Precision@K using recommend_movies."""
    ratings, movies, user_to_index, movie_to_index = load_data()

    # Load trained model
    model_path = "models/trained_model.pth"
    if not os.path.exists(model_path):
        print("Trained model not found. Please train the model first.")
        return

    model = RecommenderNN(num_users=len(user_to_index), num_movies=len(movie_to_index))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    _, test_data = train_test_split(ratings, test_size=0.3, random_state=42)
    test_users = test_data["user_index"].unique()

    precision_scores = []

    for user in test_users:
        user_data = test_data[test_data["user_index"] == user]
        if len(user_data) < 25:
            continue  # Skip users with too few ratings

        # Shuffle and split into input and evaluation sets
        user_data = user_data.sample(frac=1, random_state=42)
        input_df = user_data.iloc[:len(user_data) // 2]
        eval_df = user_data.iloc[len(user_data) // 2:]

        # Format input for recommend_movies
        input_ratings_str = ",".join(
            f"{ratings.iloc[i]['movieId']}:{ratings.iloc[i]['rating']}"
            for i in input_df.index
        )

        recommended_titles = recommend_movies(input_ratings_str, top_n=k)

        # Change 3.0 if we want a lower precision
        liked_eval_df = eval_df[eval_df["rating"] >= 3.0]
        liked_movie_ids = liked_eval_df["movieId"].tolist()
        liked_titles = movies[movies["movieId"].isin(liked_movie_ids)]["title"].tolist()

        if liked_titles:
            precision = precision_at_k(recommended_titles, liked_titles, k)
            precision_scores.append(precision)

    if precision_scores:
        avg_precision = sum(precision_scores) / len(precision_scores)
        print(f"Average Precision@{k}: {avg_precision:.4f}")
    else:
        print("No users with relevant evaluation data for Precision@K.")

if __name__ == "__main__":
    evaluate_precision_at_k(k=5)
