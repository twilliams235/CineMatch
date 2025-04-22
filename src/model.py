import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
from src.data_loader import load_data

# Load and split data
ratings, movies, user_to_index, movie_to_index = load_data()
train_data, temp_data = train_test_split(ratings, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Dataset for BPR
class BPRDataset(Dataset):
    def __init__(self, data, num_movies):
        self.user_movie_set = set(zip(data["user_index"], data["movie_index"]))
        self.users = torch.tensor(data["user_index"].values, dtype=torch.long)
        self.pos_movies = torch.tensor(data["movie_index"].values, dtype=torch.long)
        self.num_movies = num_movies

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx].item()
        pos_movie = self.pos_movies[idx].item()
        while True:
            neg_movie = random.randint(0, self.num_movies - 1)
            if (user, neg_movie) not in self.user_movie_set:
                break
        return torch.tensor(user), torch.tensor(pos_movie), torch.tensor(neg_movie)

# BPR Model
class RecommenderNN_BPR(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64, hidden_dim=128, dropout=0.3, l2_reg=1e-5):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.l2_reg = l2_reg
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

    def forward(self, user_idx, pos_movie_idx, neg_movie_idx):
        user_emb = self.user_embedding(user_idx)
        pos_emb = self.movie_embedding(pos_movie_idx)
        neg_emb = self.movie_embedding(neg_movie_idx)

        pos_input = torch.cat([user_emb, pos_emb], dim=1)
        neg_input = torch.cat([user_emb, neg_emb], dim=1)

        pos_score = self.ff(pos_input).squeeze()
        neg_score = self.ff(neg_input).squeeze()
        return pos_score, neg_score

    def l2_loss(self):
        return self.l2_reg * (
            torch.norm(self.user_embedding.weight, p=2) +
            torch.norm(self.movie_embedding.weight, p=2)
        )

# BPR loss function
def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

# Train model
def train_model(epochs=5, lr=5e-4):
    model = RecommenderNN_BPR(
        num_users=len(user_to_index),
        num_movies=len(movie_to_index)
    )
    train_loader = DataLoader(BPRDataset(train_data, len(movie_to_index)), batch_size=256, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user, pos_movie, neg_movie in train_loader:
            optimizer.zero_grad()
            pos_scores, neg_scores = model(user, pos_movie, neg_movie)
            loss = bpr_loss(pos_scores, neg_scores) + model.l2_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    validate_model(model, val_data, k=10)
    torch.save(model.state_dict(), "models/trained_model_bpr.pth")
    print("Training complete. Model saved!")

def validate_model(model, val_data, k=10):
    model.eval()
    user_groups = val_data.groupby("user_index")
    hit_count = 0
    total_users = 0

    with torch.no_grad():
        for user, group in user_groups:
            if len(group) > 100:
                continue  # skip users with too manyinteractions
            if len(group) < 30:
                continue  # skip users with very few interactions

            user_tensor = torch.tensor([user], dtype=torch.long)
            pos_movies = group.sort_values("rating", ascending=False)["movie_index"].tolist()

            input_pos = pos_movies[:len(pos_movies)//2]  # used to create user embedding
            heldout = pos_movies[len(pos_movies)//2:]    # evaluation items

            # Build user embedding from input_pos
            input_tensor = torch.tensor(input_pos, dtype=torch.long)
            user_emb = model.user_embedding(user_tensor).repeat(len(input_pos), 1)
            movie_embs = model.movie_embedding(input_tensor)
            user_vector = torch.sum(user_emb * movie_embs, dim=0) / len(input_tensor)

            # Score all movies
            all_movie_ids = torch.arange(model.movie_embedding.num_embeddings)
            all_embs = model.movie_embedding(all_movie_ids)
            scores = torch.matmul(all_embs, user_vector)

            # Remove input_pos movies from scoring
            scores[input_tensor] = -float("inf")

            # Get top-k recommendations
            top_k = torch.topk(scores, k=k).indices.tolist()

            if any(held_movie in top_k for held_movie in heldout):
                hit_count += 1
            total_users += 1

    hit_rate = hit_count / total_users if total_users > 0 else 0
    print(f"Validation Hit@{k}: {hit_rate:.4f}")


if __name__ == "__main__":
    train_model()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from src.data_loader import load_data

# # Load Data
# ratings, movies, user_to_index, movie_to_index = load_data()

# # Train-validation-test split
# train_data, temp_data = train_test_split(ratings, test_size=0.3, random_state=42)
# val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# # Define PyTorch Dataset Class
# class MovieDataset(Dataset):
#     def __init__(self, data):
#         self.users = torch.tensor(data["user_index"].values, dtype=torch.long)
#         self.movies = torch.tensor(data["movie_index"].values, dtype=torch.long)
#         self.ratings = torch.tensor(data["rating"].values, dtype=torch.float32)

#     def __len__(self):
#         return len(self.ratings)

#     def __getitem__(self, idx):
#         return self.users[idx], self.movies[idx], self.ratings[idx]

# # Loaders
# train_loader = DataLoader(MovieDataset(train_data), batch_size=256, shuffle=True)
# val_loader = DataLoader(MovieDataset(val_data), batch_size=256)
# test_loader = DataLoader(MovieDataset(test_data), batch_size=256)

# # Define PyTorch Model
# class RecommenderNN(nn.Module):
#     def __init__(self, num_users, num_movies, embedding_dim=50):
#         super(RecommenderNN, self).__init__()
#         self.user_embedding = nn.Embedding(num_users, embedding_dim)
#         self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
#         self.user_bias = nn.Embedding(num_users, 1)
#         self.movie_bias = nn.Embedding(num_movies, 1)
#         self.fc1 = nn.Linear(embedding_dim * 2, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)

#     def forward(self, user, movie):
#         user_embedded = self.user_embedding(user)
#         movie_embedded = self.movie_embedding(movie)
#         x = torch.cat([user_embedded, movie_embedded], dim=1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return (self.fc3(x).squeeze() + 
#                 self.user_bias(user).squeeze() + 
#                 self.movie_bias(movie).squeeze())

# # Initialize Model
# num_users = len(user_to_index)
# num_movies = len(movie_to_index)
# model = RecommenderNN(num_users, num_movies)

# # Training Function
# def train_model(epochs=5, lr=0.001):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0
#         for users, movies, ratings in train_loader:
#             optimizer.zero_grad()
#             predictions = model(users, movies)
#             loss = criterion(predictions, ratings)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")
    
#     print("Training complete. Evaluating on validation set...")
#     validate_model()
    
#     # Save model
#     torch.save(model.state_dict(), "models/trained_model.pth")
#     print("Model training complete & saved!")

# # Validation Function
# def validate_model():
#     model.eval()
#     val_loss = 0
#     criterion = nn.MSELoss()
#     with torch.no_grad():
#         for users, movies, ratings in val_loader:
#             predictions = model(users, movies)
#             loss = criterion(predictions, ratings)
#             val_loss += loss.item()
    
#     print(f"Validation MSE Loss: {val_loss/len(val_loader)}")

# if __name__ == "__main__":
#     train_model()