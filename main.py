import argparse
from src.model import train_model
from src.recommend import recommend_movies

def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--train", action="store_true", help="Train the recommendation model")
    parser.add_argument("--recommend", action="store_true", help="Get movie recommendations")
    args = parser.parse_args()

    if args.train:
        train_model()

    elif args.recommend:
        print("Enter watched movie IDs (comma-separated): ")
        watched_movies = list(map(int, input().split(",")))
        print("Enter ratings for those movies (comma-separated): ")
        given_ratings = list(map(float, input().split(",")))

        recommendations = recommend_movies(watched_movies, given_ratings)
        print("\nTop 5 Recommended Movies:")
        print(recommendations)

    else:
        print("Use --train to train the model or --recommend to get recommendations.")

if __name__ == "__main__":
    main()
