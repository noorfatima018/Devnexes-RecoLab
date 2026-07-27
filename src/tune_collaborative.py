from data_loader import load_ratings, load_movies, get_train_test_split
from collaborative_filtering import CollaborativeFilteringModel
from evaluation import evaluate_collaborative


def tune_hyperparameters():
    ratings = load_ratings()
    movies = load_movies()
    train, test = get_train_test_split(ratings)

    # Different combinations try karenge
    configs = [
        {"n_factors": 10, "learning_rate": 0.01, "n_epochs": 20},
        {"n_factors": 20, "learning_rate": 0.01, "n_epochs": 20},
        {"n_factors": 30, "learning_rate": 0.01, "n_epochs": 20},
        {"n_factors": 20, "learning_rate": 0.005, "n_epochs": 30},
        {"n_factors": 20, "learning_rate": 0.02, "n_epochs": 20},
    ]

    results = []

    for i, config in enumerate(configs):
        print(f"\n{'='*50}")
        print(f"Config {i+1}/{len(configs)}: {config}")
        print(f"{'='*50}")

        model = CollaborativeFilteringModel(
            n_factors=config["n_factors"],
            learning_rate=config["learning_rate"],
            n_epochs=config["n_epochs"],
        )
        model.fit(train)

        precision = evaluate_collaborative(model, train, test, movies, k=10)
        results.append({**config, "precision_at_10": precision})

        print(f"Precision@10 for this config: {precision:.4f}")

    print(f"\n{'='*50}")
    print("HYPERPARAMETER TUNING SUMMARY")
    print(f"{'='*50}")
    for r in results:
        print(r)

    best = max(results, key=lambda r: r["precision_at_10"])
    print(f"\nBest Config: {best}")

    return results, best


if __name__ == "__main__":
    results, best = tune_hyperparameters()

    # Best config ke sath final model train kar ke save karte hain
    print("\nTraining final model with best config...")
    ratings = load_ratings()
    train, test = get_train_test_split(ratings)

    final_model = CollaborativeFilteringModel(
        n_factors=best["n_factors"],
        learning_rate=best["learning_rate"],
        n_epochs=best["n_epochs"],
    )
    final_model.fit(train)
    final_model.save("models/collaborative_filtering_best.pkl")