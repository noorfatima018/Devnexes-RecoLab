import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
import requests

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.data_loader import load_ratings, load_movies, get_train_test_split
from src.baseline_model import PopularityRecommender
from src.collaborative_filtering import CollaborativeFilteringModel
from src.content_based import ContentBasedRecommender
from src.hybrid_model import HybridRecommender


app = Flask(__name__)
CORS(app)  # taake frontend (alag origin se) API ko call kar sake


GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# Poster URLs ko cache karte hain taake har request pe TMDB ko dobara call na karna pade
poster_cache = {}


def extract_title_and_year(full_title):
    """'Toy Story (1995)' se -> ('Toy Story', '1995') nikalta hai"""
    if full_title.endswith(')') and '(' in full_title:
        title_part = full_title[:full_title.rfind('(')].strip()
        year_part = full_title[full_title.rfind('(') + 1: full_title.rfind(')')].strip()
        return title_part, year_part
    return full_title, None


def fetch_poster_url(full_title):
    """TMDB se movie ka poster URL fetch karta hai (cache ke sath)"""
    if full_title in poster_cache:
        return poster_cache[full_title]

    if not TMDB_API_KEY:
        poster_cache[full_title] = None
        return None

    title, year = extract_title_and_year(full_title)

    try:
        params = {"api_key": TMDB_API_KEY, "query": title}
        if year:
            params["year"] = year

        response = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=5)
        data = response.json()

        results = data.get("results", [])
        if results and results[0].get("poster_path"):
            poster_url = TMDB_IMAGE_BASE + results[0]["poster_path"]
            poster_cache[full_title] = poster_url
            return poster_url

    except Exception as e:
        print(f"Poster fetch failed for '{full_title}': {e}")

    poster_cache[full_title] = None
    return None


print("Loading data and training models... (this happens once at startup)")
ratings = load_ratings()
movies = load_movies()
train, test = get_train_test_split(ratings)

popularity_model = PopularityRecommender()
popularity_model.fit(train)

cf_model = CollaborativeFilteringModel(n_factors=20, learning_rate=0.01, n_epochs=20)
cf_model.fit(train)

cb_model = ContentBasedRecommender()
cb_model.fit(movies)

hybrid = HybridRecommender(cf_model, cb_model, popularity_model, min_ratings_threshold=5)
print("Models ready! API is live.")

if not TMDB_API_KEY:
    print("WARNING: TMDB_API_KEY not found in .env — posters will not be shown.")


def movie_to_dict(item_id):
    """Ek movie ki poori info dictionary mein return karta hai (frontend ke liye)"""
    row = movies[movies['item_id'] == item_id].iloc[0]
    genre_indices = [i for i in range(19) if row[f'genre_{i}'] == 1]
    genre_labels = [GENRE_NAMES[i] for i in genre_indices]
    poster_url = fetch_poster_url(row['title'])

    return {
        "item_id": int(item_id),
        "title": row['title'],
        "genres": genre_labels,
        "poster_url": poster_url,
    }


@app.route('/api/users', methods=['GET'])
def get_users():
    """Saare available user IDs return karta hai (dropdown ke liye)"""
    user_ids = sorted(ratings['user_id'].unique().tolist())
    return jsonify({"users": user_ids})


@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Saare genres return karta hai (new-user onboarding ke liye)"""
    return jsonify({"genres": GENRE_NAMES[1:]})  # "unknown" skip karte hain


@app.route('/api/recommend/user/<int:user_id>', methods=['GET'])
def recommend_for_existing_user(user_id):
    """Existing user ke liye hybrid recommendations deta hai"""
    try:
        n = int(request.args.get('n', 10))
        recommended_ids, strategy = hybrid.recommend(user_id, train, movies, n=n)

        if not recommended_ids:
            return jsonify({
                "error": "No recommendations could be generated for this user.",
                "strategy": strategy
            }), 200

        recommendations = [movie_to_dict(item_id) for item_id in recommended_ids]

        return jsonify({
            "user_id": user_id,
            "strategy": strategy,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend/new-user', methods=['POST'])
def recommend_for_new_user():
    """Naye user ke liye, selected genres ke basis par recommendations deta hai"""
    try:
        data = request.get_json()
        selected_genres = data.get('genres', [])
        n = int(data.get('n', 10))

        if not selected_genres:
            return jsonify({"error": "Please select at least one genre."}), 400

        genre_indices = [GENRE_NAMES.index(g) for g in selected_genres if g in GENRE_NAMES]
        genre_cols = [f'genre_{i}' for i in genre_indices]

        scores = movies[genre_cols].sum(axis=1)
        top_indices = scores.sort_values(ascending=False).index[:n]
        recommended_ids = movies.loc[top_indices, 'item_id'].tolist()

        recommendations = [movie_to_dict(item_id) for item_id in recommended_ids]

        return jsonify({
            "strategy": "genre_preference (new user onboarding)",
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)