import argparse
from src.model import train_model
from src.recommend import recommend_movies
from src.lookup import lookup_titles

def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # --- train subcommand ---
    p_train = subparsers.add_parser("train", help="Train the recommendation model")
    p_train.add_argument("--mode", choices=["bce", "bpr"], default="bce",
                         help="Training objective: 'bce' (implicit NCF) or 'bpr' (pairwise ranking)")
    p_train.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p_train.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    p_train.add_argument("--neg-ratio", type=int, default=1,
                         help="Negatives per positive (for BCE and BPR sampling)")
    p_train.add_argument("--threshold", type=float, default=3.0,
                         help="Ratings > threshold are treated as positives for implicit labels")

    # --- recommend subcommand ---
    p_rec = subparsers.add_parser("recommend", help="Get movie recommendations")
    p_rec.add_argument("--ratings", type=str,
                       help='Inline ratings string like: "1:4.5, 50:3.0, 100:5.0". If omitted, you\'ll be prompted.')
    p_rec.add_argument("--model", type=str, default='bpr', help="Model to be used to recommend (bpr or bce)")
    p_rec.add_argument("--topn", type=int, default=5, help="Number of recommendations to return")
    p_rec.add_argument("--pos-threshold", type=float, default=3.0,
                       help="Treat inputs with rating > this threshold as positive seeds")
    
    # -- lookup subcommand --
    p_lookup = subparsers.add_parser("lookup", help="Find MovieLens movieId(s) by title")
    p_lookup.add_argument("--title", required=True, help='Title to search, e.g. "The Matrix"')
    p_lookup.add_argument("--topk", type=int, default=10, help="How many matches to show")

    args = parser.parse_args()

    if args.cmd == "train":
        train_model(
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            neg_ratio=args.neg_ratio,
            threshold=args.threshold,
        )

    elif args.cmd == "recommend":
        if args.ratings:
            input_ratings = args.ratings
        else:
            print("Enter your movie ratings in the format: movieId:rating, separated by commas.")
            print('Example: 1:4.5, 50:3.0, 100:5.0')
            input_ratings = input("Enter your ratings: ").strip()

        recs = recommend_movies(input_ratings, recommend_model=args.model, top_n=args.topn, pos_threshold=args.pos_threshold)

        if not recs:
            print("No recommendations could be generated. Provide at least one positive rating (> pos-threshold).")
            return

        print(f"\nTop {args.topn} Recommended Movies:")
        for i, title in enumerate(recs, 1):
            print(f"{i}. {title}")

    elif args.cmd == "lookup":
        matches = lookup_titles(args.title, topk=args.topk)
        if not matches:
            print("No matches found.")
        else:
            print(f"Top {len(matches)} matches:")
            for mid, title, year in matches:
                y = f" ({year})" if year else ""
                print(f"{mid:<8}  {title}{y}")

if __name__ == "__main__":
    main()
