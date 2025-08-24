import os
import torch
import pandas as pd
from typing import List
from src.data_loader import load_data
from src.model import RecommenderNN

# Load Data & Model
ratings, movies, user_to_index, movie_to_index = load_data()
num_users = len(user_to_index)
num_movies = len(movie_to_index)

model = RecommenderNN(num_users, num_movies)


def _safe_movie_index(mid: int):
    return movie_to_index[mid] if mid in movie_to_index else None

def recommend_movies(input_ratings: str, recommend_model: str, top_n: int = 5, pos_threshold: float = 3.0) -> List[str]:
    """
    Cold-start recommendation for an ad-hoc user who supplies "movieId:rating" pairs.
    We build a pseudo "user embedding" by averaging the learned movie embeddings
    of positively rated movies (> pos_threshold). Then recommend by similarity
    (dot product) against all movie embeddings. Excludes seen movies.
    """
    if recommend_model == 'bpr':
        model_path = "models/trained_model_bpr.pth"
    elif recommend_model == 'bce':
        model_path = "models/trained_model_bce.pth"
    else:
        raise ValueError("Not a valid model option")

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
    # Parse input "movieId:rating" pairs
    input_pairs = []
    for pair in input_ratings.split(","):
        pair = pair.strip()
        if not pair:
            continue
        m, r = pair.split(":")
        input_pairs.append((int(m), float(r)))

    # Filter positives (rating > threshold)
    pos_movie_ids = [m for (m, r) in input_pairs if r > pos_threshold]
    seen_movie_ids = {m for (m, _) in input_pairs}

    # Map to indices, drop unknowns
    pos_idxs = [ _safe_movie_index(m) for m in pos_movie_ids ]
    pos_idxs = [ idx for idx in pos_idxs if idx is not None ]

    if not pos_idxs:
        print("No valid positive movies found in input! Provide at least one rating > threshold.")
        return []

    with torch.no_grad():
        movie_emb = model.movie_embedding.weight  # [num_movies, E]
        # Build pseudo user embedding as mean of positive movie embeddings
        u = movie_emb[pos_idxs].mean(dim=0)       # [E]
        scores = (movie_emb @ u)                  # [num_movies]

        # Exclude seen
        seen_idxs = set(idx for idx in [ _safe_movie_index(m) for m in seen_movie_ids ] if idx is not None)
        scores_list = scores.numpy()
        # set seen to very low score
        for idx in seen_idxs:
            scores_list[idx] = -1e9

        top_idx = torch.tensor(scores_list).topk(top_n).indices.tolist()

    # Map indices back to titles via inverse index -> row lookup:
    # The original code indexed movies via iloc[idx]; here we must invert movie_to_index.
    # Build inverse map: index -> movieId
    inv_movie_index = {v: k for k, v in movie_to_index.items()}
    rec_titles = []
    for idx in top_idx:
        movie_id = inv_movie_index[idx]
        # find row in movies DataFrame where movieId == movie_id
        title_row = movies.loc[movies["movieId"] == movie_id]
        if len(title_row) > 0:
            rec_titles.append(title_row.iloc[0]["title"])
        else:
            rec_titles.append(f"movieId {movie_id}")

    return rec_titles
