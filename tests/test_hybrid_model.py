import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_ratings, load_movies, get_train_test_split
from src.baseline_model import PopularityRecommender
from src.collaborative_filtering import CollaborativeFilteringModel
from src.content_based import ContentBasedRecommender
from src.hybrid_model import HybridRecommender


def get_fitted_hybrid():
    """Test ke liye ek fitted hybrid model return karta hai (reuse karne ke liye)"""
    ratings = load_ratings()
    movies = load_movies()
    train, test = get_train_test_split(ratings)

    popularity_model = PopularityRecommender()
    popularity_model.fit(train)

    cf_model = CollaborativeFilteringModel(n_factors=10, learning_rate=0.01, n_epochs=5)
    cf_model.fit(train)  # kam epochs — test fast chalne ke liye

    cb_model = ContentBasedRecommender()
    cb_model.fit(movies)

    hybrid = HybridRecommender(cf_model, cb_model, popularity_model, min_ratings_threshold=5)

    return hybrid, train, movies


def test_new_user_gets_popularity_fallback():
    """Bilkul naye user (jiska koi rating history nahi) ko popularity fallback milna chahiye"""
    hybrid, train, movies = get_fitted_hybrid()

    fake_new_user_id = 999999
    recs, strategy = hybrid.recommend(fake_new_user_id, train, movies, n=5)

    assert "popularity" in strategy
    assert len(recs) == 5


def test_regular_user_gets_collaborative_filtering():
    """User jiske paas kaafi ratings hain, usko CF se recommendations milni chahiye"""
    hybrid, train, movies = get_fitted_hybrid()

    # Sabse zyada ratings wala user dhoondte hain
    active_user = train['user_id'].value_counts().idxmax()
    recs, strategy = hybrid.recommend(active_user, train, movies, n=5)

    assert "collaborative_filtering" in strategy
    assert len(recs) == 5


def test_new_item_gets_content_based_fallback():
    """Naya item (jiska koi rating history nahi) content-based se recommend hona chahiye"""
    hybrid, train, movies = get_fitted_hybrid()

    sample_item_id = movies['item_id'].iloc[0]
    recs, strategy = hybrid.recommend_for_new_item_scenario(sample_item_id, movies, n=5)

    assert "content_based" in strategy
    assert len(recs) <= 5


def test_recommendations_do_not_include_already_rated_items():
    """CF-based recommendations mein user ki already-rated movies nahi honi chahiye"""
    hybrid, train, movies = get_fitted_hybrid()

    active_user = train['user_id'].value_counts().idxmax()
    already_rated = set(train[train['user_id'] == active_user]['item_id'])

    recs, strategy = hybrid.recommend(active_user, train, movies, n=10)

    overlap = set(recs) & already_rated
    assert len(overlap) == 0, "Already-rated items recommendations mein nahi hone chahiye"


if __name__ == "__main__":
    test_new_user_gets_popularity_fallback()
    test_regular_user_gets_collaborative_filtering()
    test_new_item_gets_content_based_fallback()
    test_recommendations_do_not_include_already_rated_items()
    print("All hybrid model tests passed!")