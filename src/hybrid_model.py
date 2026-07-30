import numpy as np


class HybridRecommender:
    """
    Collaborative Filtering aur Content-Based Filtering ko combine karta hai.

    Switching Strategy:
    - Agar user ke paas training data mein kam ratings hain (< threshold),
      to Content-Based Filtering use karta hai (cold-start handling).
    - Agar user ke paas kaafi ratings hain, to Collaborative Filtering use
      karta hai (zyada accurate/personalized).
    - Agar user bilkul naya hai (koi rating nahi), to genre-preferences
      (agar di gayi hon) ya popularity fallback use karta hai.
    """

    def __init__(self, cf_model, cb_model, popularity_model,
                 min_ratings_threshold=5):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.popularity_model = popularity_model
        self.min_ratings_threshold = min_ratings_threshold

    def recommend(self, user_id, train_ratings, movies_df, n=10):
        """
        Ek user ke liye hybrid recommendations deta hai, uske rating count
        ke hisaab se sahi model choose kar ke.
        """
        user_ratings = train_ratings[train_ratings['user_id'] == user_id]
        num_ratings = len(user_ratings)

        if num_ratings == 0:
            # Bilkul naya user — koi rating history nahi hai
            # Fallback: popularity-based recommendations (safe default)
            return self.popularity_model.recommend(n=n), "popularity_fallback (new user, no history)"

        elif num_ratings < self.min_ratings_threshold:
            # Sparse user — Content-Based use karte hain (cold-start handling)
            user_rated_items = user_ratings['item_id'].tolist()
            recommendations = self.cb_model.recommend_for_user(user_rated_items, n=n)
            return recommendations, "content_based (sparse user, cold-start handling)"

        else:
            # Kaafi data hai — Collaborative Filtering use karte hain (zyada accurate)
            user_rated_items = set(user_ratings['item_id'].tolist())
            all_item_ids = movies_df['item_id'].unique()

            predictions = []
            for item_id in all_item_ids:
                if item_id in user_rated_items:
                    continue
                pred_rating = self.cf_model.predict(user_id, item_id)
                predictions.append((item_id, pred_rating))

            predictions.sort(key=lambda x: x[1], reverse=True)
            recommendations = [item_id for item_id, _ in predictions[:n]]
            return recommendations, "collaborative_filtering (sufficient user history)"

    def recommend_for_new_item_scenario(self, item_id, movies_df, n=10):
        """
        Cold-start ITEM scenario: agar koi movie bilkul nayi hai (kisi ne rate nahi ki),
        to Collaborative Filtering us par kaam nahi karega. Content-Based fallback
        use karte hain kyunke wo sirf genre metadata pe depend karta hai.
        """
        similar_items = self.cb_model.get_similar_items(item_id, n=n)
        return similar_items, "content_based (new item, no rating history)"


if __name__ == "__main__":
    from data_loader import load_ratings, load_movies, get_train_test_split
    from baseline_model import PopularityRecommender
    from collaborative_filtering import CollaborativeFilteringModel
    from content_based import ContentBasedRecommender

    ratings = load_ratings()
    movies = load_movies()
    train, test = get_train_test_split(ratings)

    # Teeno base models train karte hain
    print("Training baseline model...")
    popularity_model = PopularityRecommender()
    popularity_model.fit(train)

    print("Training collaborative filtering model...")
    cf_model = CollaborativeFilteringModel(n_factors=20, learning_rate=0.01, n_epochs=20)
    cf_model.fit(train)

    print("Training content-based model...")
    cb_model = ContentBasedRecommender()
    cb_model.fit(movies)

    # Hybrid model banate hain
    hybrid = HybridRecommender(cf_model, cb_model, popularity_model, min_ratings_threshold=5)

    # Scenario 1: Ek regular user (kaafi ratings wala)
    regular_user = train['user_id'].value_counts().idxmax()  # sabse zyada ratings wala user
    recs, strategy = hybrid.recommend(regular_user, train, movies, n=10)
    titles = movies.set_index('item_id').loc[recs, 'title']
    print(f"\n--- User {regular_user} (Regular, lots of history) ---")
    print(f"Strategy used: {strategy}")
    print(titles)

    # Scenario 2: Ek sparse user (kam ratings wala, agar exist kare)
    rating_counts = train['user_id'].value_counts()
    sparse_users = rating_counts[rating_counts < 5]
    if len(sparse_users) > 0:
        sparse_user = sparse_users.index[0]
        recs, strategy = hybrid.recommend(sparse_user, train, movies, n=10)
        titles = movies.set_index('item_id').loc[recs, 'title']
        print(f"\n--- User {sparse_user} (Sparse, cold-start) ---")
        print(f"Strategy used: {strategy}")
        print(titles)
    else:
        print("\nNo sparse users found in this split (dataset filtered by MovieLens).")

    # Scenario 3: Bilkul naya user (fake user_id jo dataset mein exist hi nahi karta)
    new_user_id = 99999
    recs, strategy = hybrid.recommend(new_user_id, train, movies, n=10)
    titles = movies.set_index('item_id').loc[recs, 'title']
    print(f"\n--- User {new_user_id} (Brand new, no history at all) ---")
    print(f"Strategy used: {strategy}")
    print(titles)

    # Scenario 4: Naya item (cold-start item scenario)
    sample_item = 1
    recs, strategy = hybrid.recommend_for_new_item_scenario(sample_item, movies, n=5)
    titles = movies.set_index('item_id').loc[recs, 'title']
    print(f"\n--- New Item Scenario (item_id={sample_item}) ---")
    print(f"Strategy used: {strategy}")
    print(titles)