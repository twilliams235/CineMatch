import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, Set
from src.data_loader import load_data

# ---------------------------
# Model: Neural Collaborative Filtering (MLP head)
# ---------------------------
class RecommenderNN(nn.Module):
    """
    Neural CF model: user/item embeddings -> MLP -> logit score.
    Use:
      - BCEWithLogitsLoss for implicit feedback classification
      - BPR loss on pairwise (u, i+, j-) triples with score differences
    """
    def __init__(self, num_users: int, num_movies: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        hidden = [128, 64]
        layers = []
        dim_in = 2 * embedding_dim
        for h in hidden:
            layers += [nn.Linear(dim_in, h), nn.ReLU()]
            dim_in = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dim_in, 1)  # logit

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(users)      # [B, E]
        v = self.movie_embedding(movies)    # [B, E]
        x = torch.cat([u, v], dim=1)        # [B, 2E]
        h = self.mlp(x)                     # [B, H]
        logit = self.out(h).squeeze(-1)     # [B]
        return logit

    def score(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        """Alias for clarity in BPR."""
        return self.forward(users, movies)


# ---------------------------
# Helpers for implicit feedback
# ---------------------------
def _build_user_pos_sets(df, user_col="user_index", item_col="movie_index", label_col="implicit") -> Dict[int, Set[int]]:
    user_pos: Dict[int, Set[int]] = {}
    pos_df = df[df[label_col] == 1]
    for u, i in zip(pos_df[user_col].values, pos_df[item_col].values):
        user_pos.setdefault(int(u), set()).add(int(i))
    return user_pos


# ---------------------------
# Datasets
# ---------------------------
class NCFImplicitDataset(Dataset):
    """
    Yields (user, item, label) where label in {0,1}.
    Positives: rating > threshold (precomputed "implicit" == 1).
    Negatives: for each positive, sample 'neg_ratio' random unseen items.
    Negative sampling is done on-the-fly for memory efficiency.
    """
    def __init__(self, df, num_items: int, neg_ratio: int = 1, threshold: float = 3.0):
        self.df = df
        self.num_items = num_items
        self.neg_ratio = neg_ratio

        # positives
        self.pos_df = self.df[self.df["implicit"] == 1][["user_index", "movie_index"]].reset_index(drop=True)
        self.num_pos = len(self.pos_df)

        self.user_pos = _build_user_pos_sets(self.df)

        self.length = self.num_pos * (1 + self.neg_ratio)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pos_idx = idx % self.num_pos
        u = int(self.pos_df.iloc[pos_idx]["user_index"])
        i_pos = int(self.pos_df.iloc[pos_idx]["movie_index"])

        if idx < self.num_pos:
            # positive example
            return torch.tensor(u), torch.tensor(i_pos), torch.tensor(1.0)
        else:
            # negative example: sample unseen item j
            seen = self.user_pos.get(u, set())
            j = i_pos
            while j in seen:
                j = random.randrange(self.num_items)
            return torch.tensor(u), torch.tensor(j), torch.tensor(0.0)


class BPRDataset(Dataset):
    """
    Yields (user, i_pos, j_neg) triples for BPR.
    For each positive (u, i_pos), sample an unseen j_neg on-the-fly.
    """
    def __init__(self, df, num_items: int, neg_ratio: int = 1):
        self.pos = df[df["implicit"] == 1][["user_index", "movie_index"]].reset_index(drop=True)
        self.num_items = num_items
        self.neg_ratio = neg_ratio
        self.user_pos = _build_user_pos_sets(df)
        self.length = len(self.pos) * self.neg_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        base_idx = idx % len(self.pos)
        u = int(self.pos.iloc[base_idx]["user_index"])
        i_pos = int(self.pos.iloc[base_idx]["movie_index"])

        # sample negative j not in user's positives
        seen = self.user_pos.get(u, set())
        j_neg = i_pos
        while j_neg in seen:
            j_neg = random.randrange(self.num_items)

        return torch.tensor(u), torch.tensor(i_pos), torch.tensor(j_neg)


# ---------------------------
# Training functions
# ---------------------------
def bpr_loss(model: RecommenderNN, users, i_pos, j_neg, l2: float = 1e-6):
    """
    BPR loss: -log sigma(s(u,i) - s(u,j)) + L2 regularization
    """
    s_pos = model.score(users, i_pos)
    s_neg = model.score(users, j_neg)
    diff = s_pos - s_neg
    loss = -nn.functional.logsigmoid(diff).mean()

    # L2 on embeddings
    reg = 0.0
    # grab the specific embeddings used in this batch to regularize
    u_emb = model.user_embedding(users)
    i_emb = model.movie_embedding(i_pos)
    j_emb = model.movie_embedding(j_neg)
    reg = (u_emb.pow(2).sum(dim=1) + i_emb.pow(2).sum(dim=1) + j_emb.pow(2).sum(dim=1)).mean()

    return loss + l2 * reg


def train_model(
    mode: str = "bce",
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 4096,
    neg_ratio: int = 1,
    threshold: float = 3.0,
    seed: int = 42
):
    """
    Train either:
      - mode="bce": NCF on implicit feedback with BCEWithLogitsLoss
      - mode="bpr": pairwise BPR loss
    Positives = ratings > threshold
    Negatives (for BCE) = random unseen items per positive
    Negatives (for BPR) = sampled on-the-fly j_neg
    """
    torch.manual_seed(seed)
    random.seed(seed)

    ratings, movies, user_to_index, movie_to_index = load_data(threshold=threshold)
    num_users = len(user_to_index)
    num_items = len(movie_to_index)

    train_df, val_df = train_test_split(ratings, test_size=0.1, random_state=seed)

    if mode.lower() == "bpr":
        train_ds = BPRDataset(train_df, num_items, neg_ratio=neg_ratio)
        val_ds   = BPRDataset(val_df,   num_items, neg_ratio=1)  # fewer for eval
        collate_fn = None
    else:
        train_ds = NCFImplicitDataset(train_df, num_items, neg_ratio=neg_ratio, threshold=threshold)
        val_ds   = NCFImplicitDataset(val_df,   num_items, neg_ratio=1,        threshold=threshold)
        collate_fn = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    model = RecommenderNN(num_users, num_items)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if mode.lower() == "bce":
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            if mode.lower() == "bpr":
                users, i_pos, j_neg = batch
                users = users.long()
                i_pos = i_pos.long()
                j_neg = j_neg.long()
                loss = bpr_loss(model, users, i_pos, j_neg, l2=1e-6)
            else:
                users, items, labels = batch
                users  = users.long()
                items  = items.long()
                labels = labels.float()
                logits = model(users, items)
                loss   = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            running += loss.item()

        # Simple validation (reports average batch loss under respective criterion)
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if mode.lower() == "bpr":
                    users, i_pos, j_neg = batch
                    loss_v = bpr_loss(model, users.long(), i_pos.long(), j_neg.long(), l2=1e-6)
                else:
                    users, items, labels = batch
                    logits = model(users.long(), items.long())
                    loss_v = criterion(logits, labels.float())
                val_running += loss_v.item()

        print(f"Epoch {epoch}/{epochs} | train_loss={running/len(train_loader):.4f} | val_loss={val_running/len(val_loader):.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    if mode.lower() == "bpr":
        torch.save(model.state_dict(), "models/trained_model_bpr.pth")
        print(f"Training complete ({mode}). Model saved to models/trained_model_bpr.pth")
    if mode.lower() == "bce":
        torch.save(model.state_dict(), "models/trained_model_bce.pth")
        print(f"Training complete ({mode}). Model saved to models/trained_model_bce.pth")

    return model, (ratings, movies, user_to_index, movie_to_index)
