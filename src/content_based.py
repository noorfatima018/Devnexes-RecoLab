import numpy as np
import pandas as pd


class ContentBasedRecommender:
    """
    Movie genres ka use kar ke item-to-item similarity based recommendations deta hai.
    Har movie ko ek "genre vector" (0/1 values) ke through represent karta hai,
    phir cosine similarity se sabse "similar" movies dhoondta hai.
    """

    def __init__(self):
        self.movies = None
        self.genre_matrix = None
        self.similarity_matrix = None
        self.item_id_to_index = None
        self.index_to_item_id = None

    def fit(self, movies_df):
        """
        movies_df: 'item_id', 'title', aur 'genre_0' se 'genre_18' tak columns honi chahiye
        """
        self.movies = movies_df.reset_index(drop=True)

        genre_cols = [col for col in movies_df.columns if col.startswith('genre_')]
        self.genre_matrix = self.movies[genre_cols].values.astype(float)

        # Cosine similarity manually calculate karte hain (koi extra library nahi chahiye)
        norms = np.linalg.norm(self.genre_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # divide by zero se bachne ke liye
        normalized = self.genre_matrix / norms

        self.similarity_matrix = np.dot(normalized, normalized.T)

        self.item_id_to_index = {
            item_id: idx for idx, item_id in enumerate(self.movies['item_id'])
        }
        self.index_to_item_id = {
            idx: item_id for item_id, idx in self.item_id_to_index.items()
        }

    def get_similar_items(self, item_id, n=10, exclude_items=None):
        """
        Ek diye gaye movie ke sabse zyada "similar" (genre ke hisaab se) n movies return karta hai.
        """
        if exclude_items is None:
            exclude_items = set()
        else:
            exclude_items = set(exclude_items)

        if item_id not in self.item_id_to_index:
            return []

        idx = self.item_id_to_index[item_id]
        similarity_scores = self.similarity_matrix[idx]

        similar_indices = np.argsort(similarity_scores)[::-1]

        results = []
        for sim_idx in similar_indices:
            candidate_item_id = self.index_to_item_id[sim_idx]
            if candidate_item_id == item_id:
                continue  # khud ko recommend na kare
            if candidate_item_id in exclude_items:
                continue  # already-consumed items filter
            results.append(candidate_item_id)
            if len(results) == n:
                break

        return results

    def recommend_for_user(self, user_rated_items, n=10):
        """
        User ne jo movies pasand ki (high rating di), un sabki similar
        movies collect kar ke, sabse zyada common/similar recommend karta hai.
        Cold-start ke liye useful — sirf ek movie bhi user ne rate ki ho to chalega.
        """
        if not user_rated_items:
            return []

        aggregated_scores = np.zeros(len(self.movies))
        for item_id in user_rated_items:
            if item_id not in self.item_id_to_index:
                continue
            idx = self.item_id_to_index[item_id]
            aggregated_scores += self.similarity_matrix[idx]

        similar_indices = np.argsort(aggregated_scores)[::-1]

        results = []
        exclude_set = set(user_rated_items)
        for sim_idx in similar_indices:
            candidate_item_id = self.index_to_item_id[sim_idx]
            if candidate_item_id in exclude_set:
                continue
            results.append(candidate_item_id)
            if len(results) == n:
                break

        return results


if __name__ == "__main__":
    from data_loader import load_movies

    movies = load_movies()

    model = ContentBasedRecommender()
    model.fit(movies)

    # Test: Toy Story (item_id=1) jaisi movies dhoondte hain
    sample_item_id = 1
    sample_title = movies[movies['item_id'] == sample_item_id]['title'].values[0]

    similar_ids = model.get_similar_items(sample_item_id, n=10)
    similar_titles = movies.set_index('item_id').loc[similar_ids, 'title']

    print(f"Movies similar to '{sample_title}':")
    print(similar_titles)