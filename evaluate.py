#################################################################
# Per-User Precision@K using embedding-based scoring
#################################################################
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.model import RecommenderNN

def _infer_embed_dim_from_ckpt(ckpt_path: str, default_dim: int = 64) -> int:
    """Peek at checkpoint to match embedding sizes if they differ from defaults."""
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        return sd["user_embedding.weight"].shape[1]
    except Exception:
        return default_dim

def evaluate_user_preference(recommend_model: str = 'bce', k: int = 10, pos_threshold: float = 3.0):
    """
    Evaluate Precision@K on a simple holdout:
      - For each test user with enough interactions, split their rows 50/50
      - Build a pseudo user-embedding from the POSITIVE input half (rating > threshold)
      - Score the eval half by dot product with movie embeddings
      - Precision@K = fraction of top-K predicted eval items whose rating > threshold
    """
    # Load data
    ratings, movies, user_to_index, movie_to_index = load_data(threshold=pos_threshold)

    # Load trained model (match embedding dim to checkpoint if needed)
    if recommend_model == 'bpr':
        ckpt = "models/trained_model_bpr.pth"
    elif recommend_model == 'bce':
        ckpt = "models/trained_model_bce.pth"
    else:
        raise ValueError("Not a valid model option")
    
    if not os.path.exists(ckpt):
        print("Trained model not found. Please train the model first.")
        return

    embed_dim = _infer_embed_dim_from_ckpt(ckpt, default_dim=64)
    model = RecommenderNN(num_users=len(user_to_index), num_movies=len(movie_to_index), embedding_dim=embed_dim)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    # Holdout split
    _, test_df = train_test_split(ratings, test_size=0.3, random_state=42)
    test_users = test_df["user_index"].unique()

    precisions = []
    with torch.no_grad():
        for u in test_users:
            u_df = test_df[test_df["user_index"] == u]
            if len(u_df) < 30:
                continue  # need enough interactions

            u_df = u_df.sample(frac=1.0, random_state=42)
            mid = len(u_df) // 2
            input_df = u_df.iloc[:mid]
            eval_df  = u_df.iloc[mid:]

            pos_input = input_df[input_df["rating"] > pos_threshold]
            if pos_input.empty:
                continue

            # Build pseudo user embedding as the mean of positive movie embeddings
            pos_item_idx = torch.as_tensor(pos_input["movie_index"].values, dtype=torch.long)
            u_emb = model.movie_embedding(pos_item_idx).mean(dim=0)

            # Score only the eval items
            eval_item_idx = torch.as_tensor(eval_df["movie_index"].values, dtype=torch.long)
            eval_emb = model.movie_embedding(eval_item_idx)
            scores = eval_emb @ u_emb

            topk_idx = torch.topk(scores, k=min(k, scores.numel())).indices.numpy().tolist()

            # Relevant eval items (implicit: rating > threshold)
            relevant_mask = (eval_df["rating"].values > pos_threshold)
            relevant_positions = set(np.where(relevant_mask)[0].tolist())

            hits = sum(1 for i in topk_idx if i in relevant_positions)
            denom = min(k, len(eval_df))
            if denom > 0:
                precisions.append(hits / denom)

    if precisions:
        print(f"Average Precision@{k}: {np.mean(precisions):.4f}  (users: {len(precisions)})")
    else:
        print("No eligible users found for evaluation.")

if __name__ == "__main__":
    evaluate_user_preference(k=10, pos_threshold=3.0)
