import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd

from src.data_loader import load_ratings, load_movies, get_train_test_split
from src.baseline_model import PopularityRecommender
from src.collaborative_filtering import CollaborativeFilteringModel
from src.content_based import ContentBasedRecommender
from src.hybrid_model import HybridRecommender


# ---------- Page Config ----------
st.set_page_config(page_title="RecoLab - Movie Recommender", page_icon="🎬", layout="wide")


# ---------- Load Data + Train Models (Cached) ----------
@st.cache_resource(show_spinner="Loading data and training models... (this happens once)")
def load_everything():
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

    return ratings, movies, train, hybrid


# ---------- Genre Names (for new-user onboarding) ----------
GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


# ---------- Main App ----------
def main():
    st.title("🎬 RecoLab — Hybrid Movie Recommendation Engine")
    st.caption("Devnexes AI/ML Internship Project | Noor Fatima")

    try:
        ratings, movies, train, hybrid = load_everything()
    except FileNotFoundError:
        st.error(
            "⚠️ Dataset not found. Please make sure the MovieLens dataset is placed "
            "under `data/raw/ml-100k/` before running this app."
        )
        return
    except Exception as e:
        st.error(f"⚠️ Something went wrong while loading the models: {e}")
        return

    st.divider()

    # ---------- User Selection ----------
    st.subheader("Step 1: Select a User")
    user_type = st.radio(
        "Who are you?",
        ["Existing user (pick from dataset)", "New user (no rating history)"],
        horizontal=True,
    )

    selected_user_id = None
    new_user_genres = []

    if user_type == "Existing user (pick from dataset)":
        all_user_ids = sorted(ratings['user_id'].unique())
        selected_user_id = st.selectbox("Choose a user ID:", all_user_ids)
    else:
        st.info("As a new user, we don't know your taste yet — pick a few genres you enjoy.")
        new_user_genres = st.multiselect("Select genres you like:", GENRE_NAMES[1:])
        selected_user_id = 9999999  # fake ID guaranteed not to exist in dataset

    st.divider()

    # ---------- Get Recommendations ----------
    st.subheader("Step 2: Get Recommendations")

    if st.button("🎯 Get Recommendations", type="primary"):
        with st.spinner("Finding movies for you..."):
            try:
                if user_type == "New user (no rating history)" and len(new_user_genres) > 0:
                    # Naye user ke liye, genre-based content filtering use karte hain
                    recommended_ids, strategy = recommend_by_genre_preference(
                        new_user_genres, movies, n=10
                    )
                else:
                    recommended_ids, strategy = hybrid.recommend(
                        selected_user_id, train, movies, n=10
                    )

                display_recommendations(recommended_ids, movies, strategy)

            except Exception as e:
                st.error(f"⚠️ Couldn't generate recommendations: {e}")


def recommend_by_genre_preference(selected_genres, movies_df, n=10):
    """Naye user ke liye: selected genres ke basis par movies score kar ke top-N return karta hai"""
    genre_indices = [GENRE_NAMES.index(g) for g in selected_genres]
    genre_cols = [f'genre_{i}' for i in genre_indices]

    scores = movies_df[genre_cols].sum(axis=1)
    top_indices = scores.sort_values(ascending=False).index[:n]

    return movies_df.loc[top_indices, 'item_id'].tolist(), "genre_preference (new user onboarding)"


def display_recommendations(recommended_ids, movies_df, strategy):
    """Recommendations ko cards ki tarah dikhata hai, strategy explanation ke sath"""
    if not recommended_ids:
        st.warning("No recommendations could be generated for this user. Please try different genres.")
        return

    strategy_explanations = {
        "collaborative_filtering": "📊 Based on patterns from users with similar taste to you.",
        "content_based": "🎭 Based on genres similar to movies you've rated highly.",
        "popularity_fallback": "🔥 These are generally popular, highly-rated movies (we don't have enough data about you yet).",
        "genre_preference": "🎯 Based on the genres you selected.",
    }

    matched_key = next((k for k in strategy_explanations if k in strategy), None)
    explanation = strategy_explanations.get(matched_key, "Recommended for you.")

    st.success(f"**Why these movies?** {explanation}")

    recommended_movies = movies_df[movies_df['item_id'].isin(recommended_ids)].copy()
    recommended_movies['item_id'] = pd.Categorical(
        recommended_movies['item_id'], categories=recommended_ids, ordered=True
    )
    recommended_movies = recommended_movies.sort_values('item_id')

    cols = st.columns(2)
    for idx, (_, row) in enumerate(recommended_movies.iterrows()):
        with cols[idx % 2]:
            st.markdown(f"**{idx + 1}. {row['title']}**")

            genre_indices = [i for i in range(19) if row[f'genre_{i}'] == 1]
            genre_labels = [GENRE_NAMES[i] for i in genre_indices]
            st.caption(", ".join(genre_labels) if genre_labels else "No genre info")
            st.divider()


if __name__ == "__main__":
    main()