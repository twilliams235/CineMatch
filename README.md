# CineMatch: Neural Network Movie Recommender

CineMatch is a **neural network–based movie recommender system** built using the [MovieLens 20M dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).  
It leverages **Bayesian Personalized Ranking (BPR)** with implicit feedback to generate personalized movie recommendations based on the movies you’ve already watched and rated.

---

## Features
- Uses the **MovieLens 20M dataset** for large-scale recommendation training  
- **Neural network with embeddings** for users and movies  
- **Bayesian Personalized Ranking (BPR) loss** for implicit feedback learning  
- **L2 regularization + dropout** to prevent overfitting  
- Supports **genre-based filtering** (e.g., only recommend Comedies or Action films)  
- Interactive CLI for:
  - Training the model
  - Getting top-K movie recommendations  
  - Filtering results by genre

---

## Project Structure
src/
- model.py # BPR-based recommender model + training loop
- recommend.py # Movie recommendation logic
- data_loader.py # Data loading, preprocessing, and genre utilities

models/
- trained_model_bpr.pth # Saved trained model (after training)

main.py # CLI entry point

README.md


## Installation
1. Clone the repository:
   ```git clone https://github.com/your-username/CineMatch.git```
   ```cd CineMatch```
2. Install dependencies:
    ```pip install torch pandas scikit-learn```
3. Download the MovieLens 20M dataset
    ```python -m movielens_download.py```

## Training the Model
Run:
    ```python -m src.main --train```
- Trains the BPR recommender on MovieLens ratings
- Saves the model to models/trained_model_bpr.pth

## Getting Recommendations
Run:
    ```python -m src.main --recommend --top_k 5 --genre Comedy```

You’ll be prompted to enter your watched movies and ratings in the format:
    1:4.5, 50:3.0, 100:5.0

The system will:
- Build a user embedding from your rated movies
- Recommend Top-K movies you haven’t rated yet
- Optionally filter them by genre (e.g., Comedy, Action, Drama)

Example output:

Top 5 Recommended Movies in Comedy genre:
1. Groundhog Day (1993)
2. Superbad (2007)
3. The Big Lebowski (1998)
...

## Evaluation
During training, CineMatch evaluates recommendations using Hit@K:
- Splits user interactions into "training" and "held-out" items
- Checks whether at least one held-out movie appears in the top-K recommended list
- Prints the Validation Hit@10 score after training

## Future Improvements
- Extend evaluation metrics (NDCG, Precision@K, Recall@K)
- Build a web interface for easier interaction