# CineMatch: Neural Network Movie Recommender

CineMatch is a **neural network–based movie recommender system** built using the [MovieLens 20M dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).  
It supports training with both **Bayesian Personalized Ranking (BPR)** and **binary cross-entropy (BCE)** for implicit feedback, enabling flexible recommendation learning

---

## Features
- Trains on the MovieLens 20M dataset for large-scale recommendations
- User and movie embeddings learned jointly with a neural network
- Two training objectives:
    - Neural Collaborative Filtering (NCF) with BCE on implicit feedback
    - Bayesian Personalized Ranking (BPR) for pairwise ranking
- Implicit feedback handling:
    - Ratings > 3 counted as positives
    - Random unseen movies sampled as negatives
- Interactive CLI for:
    - Training the model (bce or bpr)
    - Getting top-K movie recommendations from your own ratings

---

## Project Structure
- src/
    - model.py              --> NCF + BPR recommender model and training loops
    - recommend.py          --> Movie recommendation logic
    - data_loader.py        --> Data loading, preprocessing
    - lookup.py             --> Data loading, preprocessing

- models/
    - trained_model_bpr.pth --> Saved trained model after training

- main.py                 --> CLI entry point
- EDA.py                  --> Analysis on the Kaggle dataset
- movielens_download.py   --> Script to download the dataset
- README.md


## Installation
1. Clone the repository:
   ```git clone https://github.com/your-username/CineMatch.git```
   ```cd CineMatch```
2. Install dependencies:
    ```pip install -r requirements.txt```
3. Download the MovieLens 20M dataset
    ```python -m movielens_download.py```

## Training the Model
Use the CLI train command:
- Train with NCF (BCE implicit feedback)
    ```python main.py train --mode bce --epochs 5 --neg-ratio 1```
    - Saves the model to models/trained_model_bce.pth
- Trains with BPR:
    ```python main.py train --mode bpr --epochs 5 --neg-ratio 1```
    - Saves the model to models/trained_model_bpr.pth

## Getting Recommendations
Run:
    ```python main.py recommend --ratings "1:4.5, 50:3.0, 100:5.0" --model='bpr' --topn 5 --pos-threshold 3.0```

- Ratings > `pos-threshold` (default 3.0) are treated as positive samples
- Model allows you to select which model you want to recommend with
- If `--ratings` is omitted, you'll be prompted to enter ratings interactively
- Recommendations exclude movies you already rated
- Example output:

Top 5 Recommended Movies in Comedy genre:
1. Groundhog Day (1993)
2. Superbad (2007)
3. The Big Lebowski (1998)
...

## Evaluation
During training, validation is performed on held-out data
- For BCE, validation loss = binary cross-entropy on implicit labels
- For BPR, validation loss = pairwise ranking loss

After training, CineMatch evaluates recommendations using Precision@K:
- Splits user interactions into "training" and "held-out" items
- From the held-out movies, predicts the top 10 (User has a minimum of 30 movies recommended)
- Precision@K evaluates how many of the top-K recommended movies are actually relevent
(movies the user rated above the 3.0 threshold)


**BPR Model: Average Precision@10: 0.76 (52,622 users)**

**BCE Model: Average Precision@10: 0.7564 (52,622 users)**

## Future Improvements
- Extend evaluation metrics (NDCG, Recall@K)
- Build a web interface for easier interaction