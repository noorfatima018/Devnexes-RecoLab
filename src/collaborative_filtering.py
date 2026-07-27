import numpy as np
import pandas as pd


class CollaborativeFilteringModel:
    """
    Matrix Factorization based Collaborative Filtering, implemented
    from scratch using Stochastic Gradient Descent (SGD).

    Users aur movies ko latent factor vectors mein represent karta hai;
    training ke dauran ye vectors is tarah adjust hote hain ke
    predicted rating (dot product) actual rating ke jitna qareeb ho.
    """
    def save(self, filepath):
        """Model ko file mein save karta hai (numpy format mein)"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Saved model ko wapis load karta hai"""
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model

    def __init__(self, n_factors=20, learning_rate=0.01, regularization=0.02,
                 n_epochs=20, random_state=42):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None

        self.user_id_map = None
        self.item_id_map = None
        self.reverse_item_map = None

    def fit(self, ratings_df):
        rng = np.random.RandomState(self.random_state)

        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['item_id'].unique()

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}

        n_users = len(unique_users)
        n_items = len(unique_items)

        self.user_factors = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = ratings_df['rating'].mean()

        user_indices = ratings_df['user_id'].map(self.user_id_map).values
        item_indices = ratings_df['item_id'].map(self.item_id_map).values
        rating_values = ratings_df['rating'].values

        for epoch in range(self.n_epochs):
            total_error = 0
            for u, i, r in zip(user_indices, item_indices, rating_values):
                pred = (self.global_mean + self.user_bias[u] + self.item_bias[i] +
                        np.dot(self.user_factors[u], self.item_factors[i]))
                error = r - pred

                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])

                uf_old = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (error * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (error * uf_old - self.reg * self.item_factors[i])

                total_error += error ** 2

            rmse = np.sqrt(total_error / len(rating_values))
            print(f"Epoch {epoch + 1}/{self.n_epochs} - Training RMSE: {rmse:.4f}")

    def predict(self, user_id, item_id):
        """Ek user-item pair ke liye predicted rating return karta hai"""
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return self.global_mean  # cold-start fallback

        u = self.user_id_map[user_id]
        i = self.item_id_map[item_id]

        pred = (self.global_mean + self.user_bias[u] + self.item_bias[i] +
                np.dot(self.user_factors[u], self.item_factors[i]))
        return np.clip(pred, 1, 5)

    def recommend_for_user(self, user_id, movies_df, rated_items=None, n=10):
        """Ek user ke liye top-N movies recommend karta hai"""
        if rated_items is None:
            rated_items = set()

        all_item_ids = movies_df['item_id'].unique()
        predictions = []

        for item_id in all_item_ids:
            if item_id in rated_items:
                continue
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]

        return [item_id for item_id, _ in top_n]



if __name__ == "__main__":
    from data_loader import load_ratings, load_movies, get_train_test_split

    ratings = load_ratings()
    movies = load_movies()

    train, test = get_train_test_split(ratings)

    model = CollaborativeFilteringModel(n_factors=20, n_epochs=20)
    model.fit(train)

    sample_user = 1
    user_rated = set(train[train['user_id'] == sample_user]['item_id'])

    recommended_ids = model.recommend_for_user(sample_user, movies, rated_items=user_rated, n=10)
    recommended_titles = movies.set_index('item_id').loc[recommended_ids, 'title']

    print(f"\nTop 10 Recommendations for User {sample_user}:")
    print(recommended_titles)