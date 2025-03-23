import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the dataset path
dataset_path = "/Users/tylerwilliams/.cache/kagglehub/datasets/grouplens/movielens-20m-dataset/versions/1"

# Load the ratings dataset
ratings = pd.read_csv(os.path.join(dataset_path, "rating.csv"))

# Load the movies dataset
movies = pd.read_csv(os.path.join(dataset_path, "movie.csv"))

# Display first few rows
ratings.head(), movies.head()

####################################################################
# Ratings per User
####################################################################

# ratings_per_user = ratings.groupby("userId")["rating"].count()

# max_ratings = ratings_per_user.max()
# max_user = ratings_per_user.idxmax()

# users_with_20_ratings = (ratings_per_user == 20).sum()

# print(f"Number of users with exactly 20 ratings: {users_with_20_ratings}")

# print(f"User {max_user} has the highest number of ratings: {max_ratings}")

# plt.figure(figsize=(8,5))
# sns.histplot(ratings_per_user, bins=100, kde=True)
# plt.xlabel("Number of Ratings Per User")
# plt.ylabel("Count")
# plt.title("Distribution of Ratings Per User")
# plt.xscale("log")  # Log scale for better visualization
# plt.xlim(10, 10**3)  # Limit x-axis to 10^3
# plt.grid(True)
# plt.show()

####################################################################
# Most Common Movie Genres
####################################################################

# Split genres and count occurrences
from collections import Counter

all_genres = movies['genres'].dropna().str.split('|').explode()
genre_counts = Counter(all_genres)

genre_df = pd.DataFrame(genre_counts.items(), columns=["Genre", "Count"]).sort_values(by="Count", ascending=False)

genre_df["Percentage"] = (genre_df["Count"] / genre_df["Count"].sum()) * 100

plt.figure(figsize=(8,8))
plt.pie(genre_df["Percentage"], labels=genre_df["Genre"], autopct='%1.1f%%', startangle=140, pctdistance=0.85)
plt.title("Distribution of Movie Genres")
plt.show()

####################################################################
# Most Rated Movies
####################################################################

# # Merge ratings with movie titles
# movie_ratings = ratings.merge(movies, on="movieId")

# # Calculate average rating and number of ratings per movie
# movie_stats = movie_ratings.groupby("title").agg({"rating": ["mean", "count"]})

# # Rename columns
# movie_stats.columns = ["average_rating", "num_ratings"]

# # Plot scatter plot
# plt.figure(figsize=(10,5))
# sns.scatterplot(x=movie_stats["num_ratings"], y=movie_stats["average_rating"], alpha=0.5)
# plt.xlabel("Number of Ratings")
# plt.ylabel("Average Rating")
# plt.title("Average Rating vs. Number of Ratings")
# plt.xscale("log")  # Log scale for better visualization
# plt.show()

####################################################################
# Most Rated Movies
####################################################################

# # Merge ratings with movie titles
# movie_ratings = ratings.merge(movies, on="movieId")

# # Count number of ratings per movie
# top_movies = movie_ratings.groupby("title")['rating'].count().sort_values(ascending=False).head(10)

# # Plot
# plt.figure(figsize=(10,5))
# sns.barplot(x=top_movies.values, y=top_movies.index)
# plt.xlabel("Number of Ratings")
# plt.ylabel("Movie Title")
# plt.title("Top 10 Most Rated Movies")
# plt.show()

####################################################################
# Movie Ratings Distribution
####################################################################

# plt.figure(figsize=(8,5))
# sns.histplot(ratings['rating'], bins=10)
# plt.xlabel("Movie Rating")
# plt.ylabel("Count")
# plt.title("Distribution of Movie Ratings")
# plt.show()

