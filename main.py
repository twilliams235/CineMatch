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
        print("Enter your movie ratings in the format: movieId:rating, separated by commas.")
        print("Example: 1:4.5, 50:3.0, 100:5.0")
        input_ratings = input("Enter your ratings: ")
        
        recommendations = recommend_movies(input_ratings)
        
        print("\nTop 5 Recommended Movies:")
        for idx, movie in enumerate(recommendations, start=1):
            print(f"{idx}. {movie}")

    else:
        print("Use --train to train the model or --recommend to get recommendations.")

if __name__ == "__main__":
    main()
