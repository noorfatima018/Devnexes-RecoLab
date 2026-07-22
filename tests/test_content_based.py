import sys
import os

# Project root ko path mein add karte hain taake 'src' import ho sake
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_movies
from src.content_based import ContentBasedRecommender


def get_fitted_model():
    """Test ke liye ek fitted model return karta hai (reuse karne ke liye)"""
    movies = load_movies()
    model = ContentBasedRecommender()
    model.fit(movies)
    return model, movies


def test_no_duplicates_in_recommendations():
    """Recommendations mein koi item repeat na ho"""
    model, movies = get_fitted_model()
    recommendations = model.get_similar_items(item_id=1, n=10)

    assert len(recommendations) == len(set(recommendations)), \
        "Recommendations mein duplicate items nahi hone chahiye"


def test_item_does_not_recommend_itself():
    """Movie khud ko recommend na kare"""
    model, movies = get_fitted_model()
    sample_item_id = 1

    recommendations = model.get_similar_items(item_id=sample_item_id, n=10)

    assert sample_item_id not in recommendations, \
        "Movie khud ko apni hi recommendation list mein nahi honi chahiye"


def test_consumed_items_are_filtered():
    """User ne jo movies already dekh li hain, wo dobara recommend na hon"""
    model, movies = get_fitted_model()

    already_watched = {1, 95, 422}  # Toy Story, Aladdin, aur ek Aladdin sequel

    recommendations = model.get_similar_items(
        item_id=1, n=10, exclude_items=already_watched
    )

    overlap = set(recommendations) & already_watched
    assert len(overlap) == 0, \
        "Already-watched items recommendations mein nahi aane chahiye"


def test_recommend_for_user_excludes_rated_items():
    """User-level recommendation bhi already-rated items exclude kare"""
    model, movies = get_fitted_model()

    user_rated = [1, 95, 172]  # kuch movies jo user ne rate ki
    recommendations = model.recommend_for_user(user_rated, n=10)

    overlap = set(recommendations) & set(user_rated)
    assert len(overlap) == 0, \
        "User ki apni rated movies recommendations mein nahi aani chahiye"


def test_recommendations_return_correct_count():
    """Agar n=5 manga hai to result mein 5 (ya kam, agar itni movies na hon) items hon"""
    model, movies = get_fitted_model()
    recommendations = model.get_similar_items(item_id=1, n=5)

    assert len(recommendations) <= 5


if __name__ == "__main__":
    test_no_duplicates_in_recommendations()
    test_item_does_not_recommend_itself()
    test_consumed_items_are_filtered()
    test_recommend_for_user_excludes_rated_items()
    test_recommendations_return_correct_count()
    print("All tests passed!")