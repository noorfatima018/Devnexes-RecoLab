def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Precision@K: top-K recommendations mein se kitne % actually relevant the.

    recommended_items: model ne jo top-K items recommend kiye (list)
    relevant_items: user ne actually jo items pasand kiye (list/set)
    """
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / k


def evaluate_baseline(model, test_ratings, k=10, rating_threshold=4):
    """
    Baseline model ko test set par evaluate karta hai.
    'Relevant' items wo hain jinhe user ne threshold se zyada rating di.
    Har user ke liye precision@k calculate kar ke average nikalta hai.
    """
    recommended_items = model.recommend(k)

    precisions = []
    for user_id, group in test_ratings.groupby('user_id'):
        relevant_items = group[group['rating'] >= rating_threshold]['item_id'].tolist()

        if len(relevant_items) == 0:
            continue  # is user ke koi relevant item nahi, skip karein

        precision = precision_at_k(recommended_items, relevant_items, k)
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    return avg_precision


def evaluate_collaborative(model, train_ratings, test_ratings, movies_df, k=10, rating_threshold=4):
    """
    Collaborative filtering model ko test set par evaluate karta hai.
    Har user ke liye alag recommendations generate karta hai (personalized),
    phir unka precision@k nikal kar average leta hai.
    """
    precisions = []

    test_users = test_ratings['user_id'].unique()

    for user_id in test_users:
        relevant_items = test_ratings[
            (test_ratings['user_id'] == user_id) &
            (test_ratings['rating'] >= rating_threshold)
        ]['item_id'].tolist()

        if len(relevant_items) == 0:
            continue

        user_rated_in_train = set(
            train_ratings[train_ratings['user_id'] == user_id]['item_id']
        )

        recommended_items = model.recommend_for_user(
            user_id, movies_df, rated_items=user_rated_in_train, n=k
        )

        precision = precision_at_k(recommended_items, relevant_items, k)
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    return avg_precision


if __name__ == "__main__":
    from data_loader import load_ratings, load_movies, get_train_test_split
    from baseline_model import PopularityRecommender
    from collaborative_filtering import CollaborativeFilteringModel

    ratings = load_ratings()
    movies = load_movies()
    train, test = get_train_test_split(ratings)

    # Baseline
    baseline = PopularityRecommender()
    baseline.fit(train)
    baseline_precision = evaluate_baseline(baseline, test, k=10)

    # Collaborative Filtering
    cf_model = CollaborativeFilteringModel(n_factors=20, n_epochs=20)
    cf_model.fit(train)
    cf_precision = evaluate_collaborative(cf_model, train, test, movies, k=10)

    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Baseline (Popularity)   - Precision@10: {baseline_precision:.4f}")
    print(f"Collaborative Filtering - Precision@10: {cf_precision:.4f}")