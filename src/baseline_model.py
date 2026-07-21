import pandas as pd


class PopularityRecommender:
    """
    Simple baseline model: sabse zyada aur best-rated movies recommend karta hai
    sabko (personalization nahi hoti, sirf overall popularity par based hai).
    """

    def __init__(self):
        self.popular_items = None

    def fit(self, train_ratings):
        # Har movie ka average rating aur kitni baar rate hui, calculate karein
        stats = train_ratings.groupby('item_id')['rating'].agg(['mean', 'count'])

        # Weighted score use kar rahe hain (IMDB jaisa formula) taake
        # sirf 1 rating wali movie (jo 5-star ho) top pe na aa jaye
        C = stats['mean'].mean()          # overall average rating
        m = stats['count'].quantile(0.7)  # minimum votes threshold

        stats['weighted_score'] = (
            (stats['count'] / (stats['count'] + m)) * stats['mean'] +
            (m / (stats['count'] + m)) * C
        )

        self.popular_items = stats.sort_values('weighted_score', ascending=False)

    def recommend(self, n=10):
        """Top-N popular movies ki item_id list return karta hai"""
        if self.popular_items is None:
            raise ValueError("Model abhi fit nahi hua. Pehle .fit() call karein.")
        return self.popular_items.head(n).index.tolist()


if __name__ == "__main__":
    from data_loader import load_ratings, load_movies, get_train_test_split

    ratings = load_ratings()
    movies = load_movies()

    train, test = get_train_test_split(ratings)

    model = PopularityRecommender()
    model.fit(train)

    top_10_ids = model.recommend(10)
    top_10_titles = movies.set_index('item_id').loc[top_10_ids, 'title']

    print("Top 10 Recommended Movies (Baseline - Popularity Based):")
    print(top_10_titles)