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


if __name__ == "__main__":
    from data_loader import load_ratings, get_train_test_split
    from baseline_model import PopularityRecommender

    ratings = load_ratings()
    train, test = get_train_test_split(ratings)

    model = PopularityRecommender()
    model.fit(train)

    precision = evaluate_baseline(model, test, k=10)
    print(f"Baseline Model - Average Precision@10: {precision:.4f}")
    