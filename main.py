import argparse
from src.model import train_model
from src.recommend import recommend_movies
from src.data_loader import get_all_genres, is_valid_genre

def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--train", action="store_true", help="Train the recommendation model")
    parser.add_argument("--recommend", action="store_true", help="Get movie recommendations")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top recommendations to return (default: 5)")
    parser.add_argument("--genre", type=str, help="Optional genre to filter recommendations (e.g., Comedy, Action)")
    args = parser.parse_args()

    if args.train:
        train_model()

    elif args.recommend:
        # Check genre validity if provided
        if args.genre:
            if not is_valid_genre(args.genre):
                print(f"\nError: '{args.genre}' is not a valid genre.")
                print("\nAvailable genres:")
                for genre in sorted(get_all_genres()):
                    print(f"- {genre}")
                return

        print("\nEnter your movie ratings in the format: movieId:rating, separated by commas.")
        print("Example: 1:4.5, 50:3.0, 100:5.0")
        input_ratings = input("Enter your ratings: ")
        
        genre_msg = f" in {args.genre} genre" if args.genre else ""
        print(f"\nTop {args.top_k} Recommended Movies{genre_msg}:")
        recommendations = recommend_movies(input_ratings, top_k=args.top_k, genre_filter=args.genre)
        
        for idx, movie in enumerate(recommendations, start=1):
            print(f"{idx}. {movie}")

    else:
        print("Use --train to train the model or --recommend to get recommendations.")
        print("Optional: Use --genre GENRE to filter recommendations by genre")
        print("Optional: Use --top_k NUMBER to change the number of recommendations")

if __name__ == "__main__":
    main()