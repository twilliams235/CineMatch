import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.data_loader import load_data

# Load Data
ratings, movies, user_to_index, movie_to_index = load_data()

# Train-validation-test split
train_data, temp_data = train_test_split(ratings, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Define PyTorch Dataset Class
class MovieDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data["user_index"].values, dtype=torch.long)
        self.movies = torch.tensor(data["movie_index"].values, dtype=torch.long)
        self.ratings = torch.tensor(data["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# Loaders
train_loader = DataLoader(MovieDataset(train_data), batch_size=256, shuffle=True)
val_loader = DataLoader(MovieDataset(val_data), batch_size=256)
test_loader = DataLoader(MovieDataset(test_data), batch_size=256)

# Define PyTorch Model
class RecommenderNN(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super(RecommenderNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user, movie):
        user_embedded = self.user_embedding(user)
        movie_embedded = self.movie_embedding(movie)
        x = torch.cat([user_embedded, movie_embedded], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize Model
num_users = len(user_to_index)
num_movies = len(movie_to_index)
model = RecommenderNN(num_users, num_movies)

# Training Function
def train_model(epochs=5, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for users, movies, ratings in train_loader:
            optimizer.zero_grad()
            predictions = model(users, movies).squeeze()
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")
    
    print("Training complete. Evaluating on validation set...")
    validate_model()
    
    # Save model
    torch.save(model.state_dict(), "models/trained_model.pth")
    print("Model training complete & saved!")

# Validation Function
def validate_model():
    model.eval()
    val_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for users, movies, ratings in val_loader:
            predictions = model(users, movies).squeeze()
            loss = criterion(predictions, ratings)
            val_loss += loss.item()
    
    print(f"Validation MSE Loss: {val_loss/len(val_loader)}")

if __name__ == "__main__":
    train_model()

