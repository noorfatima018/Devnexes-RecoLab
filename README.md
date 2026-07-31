# Devnexes RecoLab — Hybrid Recommendation Engine with Cold-Start Handling

## Problem Statement
Recommendation systems power modern platforms (Netflix, Amazon, Spotify) by helping users
discover relevant items. A major challenge in these systems is the **cold-start problem** —
providing good recommendations for new users or new items with little to no interaction history.
This project builds a hybrid recommendation engine that combines collaborative filtering and
content-based filtering to address this challenge.

## Objectives
- Build a baseline popularity-based recommender
- Implement collaborative filtering (matrix factorization)
- Implement content-based filtering using item metadata
- Combine both into a hybrid model to handle cold-start scenarios
- Evaluate all models using ranking-based metrics (Precision@K, Recall@K)

## Dataset
- **MovieLens 100K** — 943 users, 1,682 movies, 100,000 ratings
- Source: [GroupLens Research](https://grouplens.org/datasets/movielens/100k/)
- Public, licensed dataset for non-commercial research use

## Features
- [x] Data loading and preprocessing pipeline
- [x] Exploratory Data Analysis (sparsity, rating distributions, popularity trends)
- [x] Baseline popularity-based recommender
- [x] Evaluation framework (Precision@K)
- [x] Collaborative filtering (SGD-based matrix factorization, implemented from scratch)
- [x] Content-based filtering (genre-based cosine similarity)
- [x] Hybrid model with cold-start handling (switching strategy)
- [x] Interactive web interface (Flask API + HTML/CSS/JS frontend with TMDB poster integration)

## Technology Stack
- **Language:** Python 3.14
- **Data Processing:** pandas, numpy
- **ML/Recommendation:** scikit-learn (scikit-surprise attempted but blocked — see Limitations)
- **Visualization:** matplotlib, seaborn
- **Testing:** pytest
- **Backend API:** Flask, Flask-CORS
- **Frontend:** HTML, CSS, JavaScript (vanilla, no framework)
- **External API:** TMDB (The Movie Database) for movie posters
- **Version Control:** Git + GitHub

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/noorfatima018/Devnexes-RecoLab.git
cd Devnexes-RecoLab
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

3. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

4. Download the MovieLens 100K dataset from [GroupLens](https://grouplens.org/datasets/movielens/100k/)
   and place it under `data/raw/ml-100k/`

5. Run the baseline model:
```bash
python -m src.baseline_model
```

6. Run collaborative filtering model:
```bash
python -m src.collaborative_filtering
```

7. Run content-based filtering model:
```bash
python -m src.content_based
```

8. Run full evaluation (compares baseline and collaborative filtering):
```bash
python -m src.evaluation
```

9. Run automated tests:
```bash
python -m pytest tests/ -v
```

10. Create a `.env` file in the project root with your TMDB API key:

(Get a free key at https://www.themoviedb.org/settings/api)

11. Run the backend API (in one terminal):
```bash
python -m flask --app app.backend.api run --port 5000
```

12. Open `app/frontend/index.html` in your browser (with the backend running)

## Results So Far

**Model Comparison (Precision@10):**

| Model | Precision@10 | Notes |
|---|---|---|
| Baseline (Popularity-Based) | 0.0625 | Same recommendations for every user |
| Collaborative Filtering (SGD Matrix Factorization) | **0.0810** | Personalized per-user recommendations |

Collaborative Filtering improved over the baseline by ~30%, confirming that
personalization adds meaningful value over simple popularity-based recommendations.

**Baseline Model:**
- Top recommended movies: Star Wars (1977), Schindler's List, Shawshank Redemption, Godfather
- Same list recommended to every user (no personalization)

**Collaborative Filtering Model:**
- Implemented from scratch using Stochastic Gradient Descent (SGD)-based matrix factorization
  (built manually with NumPy due to a system-level restriction blocking the `scikit-surprise`
  library's compiled binaries)
- Learns latent factors for users and items from rating patterns
- Training RMSE decreased steadily from 0.84 to 0.73 over 20 epochs, indicating stable learning
- Produces genuinely personalized recommendations (verified manually — different users
  receive distinctly different top-10 lists)

**Hyperparameter Tuning:**

Five configurations were tested by varying `n_factors` and `learning_rate`:

| n_factors | learning_rate | n_epochs | Precision@10 |
|---|---|---|---|
| 10 | 0.01 | 20 | 0.0728 |
| **20** | **0.01** | **20** | **0.0810** (best) |
| 30 | 0.01 | 20 | 0.0766 |
| 20 | 0.005 | 30 | 0.0793 |
| 20 | 0.02 | 20 | 0.0648 |

The best configuration (`n_factors=20, learning_rate=0.01, n_epochs=20`) matched the original
default setup. Too few factors (10) underfit the data, too many (30) showed signs of
overfitting, and a higher learning rate (0.02) caused unstable convergence. The final tuned
model is saved as an artifact at `models/collaborative_filtering_best.pkl` for reuse without
retraining.

**Content-Based Filtering Model:**
- Uses movie genre metadata to compute item-to-item similarity via cosine similarity
- Recommends movies similar to ones a user has rated highly, based purely on genre overlap
- Does not require any user rating history to make sense of an item — useful for
  cold-start scenarios where collaborative filtering has no signal
- Verified manually: recommendations for "Toy Story (1995)" correctly returned other
  family/animated/comedy titles (e.g. Aladdin, Home Alone, Goofy Movie)
- Covered by automated tests confirming no duplicate recommendations, no self-recommendation,
  and correct filtering of already-consumed items

**Hybrid Model:**

Combines all three models using a switching strategy based on how much rating history a
user has:

| User Type | Ratings Count | Strategy Used |
|---|---|---|
| Brand-new user (no ratings) | 0 | Popularity fallback |
| Sparse user | < 5 | Content-based filtering |
| Regular user | ≥ 5 | Collaborative filtering |

For new items with no rating history, the model falls back to content-based similarity,
since collaborative filtering has no signal for items no one has rated yet.

**Verified scenarios:**
- Regular user (405, active history) → Collaborative Filtering → personalized recommendations
- Brand-new user (no history) → Popularity fallback → safe, broadly-liked recommendations
- New item (Toy Story) → Content-based → genre-similar movies (Aladdin, Goofy Movie, etc.)
- Note: MovieLens 100K only includes users with ≥20 ratings by design, so no naturally
  "sparse" users (< 5 ratings) exist in this dataset — this fallback path was verified
  using a simulated new-user scenario instead.

Covered by automated tests (pytest) verifying correct strategy selection per scenario and
confirming no already-rated items appear in recommendations.

**Web Interface:**
- Built a custom Flask REST API (`app/backend/api.py`) exposing endpoints for existing-user
  recommendations, new-user genre-based recommendations, and metadata (users, genres)
- Integrated the TMDB API to fetch real movie posters by title, with in-memory caching to
  avoid redundant API calls
- Built a custom frontend (`app/frontend/`) using HTML/CSS/JavaScript — no framework
  dependencies — featuring:
  - Existing user vs. new user selection flow
  - Genre-based onboarding for new users (cold-start UX)
  - Real movie posters (with a genre-colored gradient fallback if a poster isn't found)
  - Explanation banner showing which strategy (collaborative filtering, content-based,
    popularity fallback, or genre preference) produced each recommendation set
  - Loading and error states for a professional, non-technical user experience

**Dataset Insights:**
- Sparsity: ~93.7% of user-movie combinations have no rating
- Long-tail distribution: majority of movies have very few ratings — highlighting the
  cold-start challenge this project aims to solve

## Error Analysis

An error analysis was performed on the Collaborative Filtering model's test-set predictions.

**Finding:** The model's worst predictions consistently follow one pattern — it over-predicts
ratings (4.5–5) for movies that are generally well-liked in the training data (e.g. Shawshank
Redemption, Ben-Hur, Annie Hall), even when a specific test user rated them very low (1).

**Root Cause:** This reflects a cold-start / sparse-data limitation. For users with limited
rating history, the model has insufficient signal to learn their individual taste, so its
predictions drift toward the item's general popularity rather than the user's specific
preference.

**Example Failed Predictions:**

| User | Movie | Actual Rating | Predicted Rating | Error |
|---|---|---|---|---|
| 1 | Babe (1995) | 1 | 5.00 | 4.00 |
| 405 | Another Stakeout (1993) | 5 | 1.00 | 4.00 |
| 312 | Die Hard (1988) | 1 | 4.98 | 3.98 |
| 38 | Ben-Hur (1959) | 1 | 4.85 | 3.85 |

**Implication:** This finding directly motivates the hybrid model — combining collaborative
filtering with content-based filtering should help the system make better predictions for
users with sparse rating history, rather than defaulting to general popularity.

## Testing Notes
- Manual verification of data loading against known MovieLens statistics (943 users, 1682 movies, 100000 ratings) — confirmed match
- Baseline model output manually reviewed for sanity (recommends well-known, highly-rated films)
- Collaborative filtering training RMSE monitored across epochs to confirm stable convergence
- Manually verified that different users receive different recommendation lists (confirms personalization is working)
- Automated tests (pytest) for content-based filtering: no duplicate recommendations,
  no self-recommendation, and correct exclusion of already-consumed items — all passing
- Automated tests (pytest) for the hybrid model: correct strategy selection per user type,
  and no already-rated items appearing in recommendations — all passing
- Manually tested the web interface end-to-end for both existing-user and new-user flows,
  including error handling when the backend is unreachable

## Limitations (Current Stage)
- `scikit-surprise` library could not be used due to a system-level Application Control
  policy blocking its compiled binaries — collaborative filtering was implemented manually
  using NumPy as a result
- Content-based filtering currently uses genre data only; does not yet incorporate other
  metadata (e.g. cast, director, release year)
- Evaluation currently limited to Precision@K; additional metrics (NDCG, Recall@K) planned
- MovieLens 100K only includes users with ≥20 ratings by design, so the "sparse user"
  fallback path in the hybrid model could not be verified with a naturally-occurring user
- Poster fetching depends on the TMDB API being available and an API key being configured;
  the app falls back to genre-colored placeholder cards if a poster can't be found

## Future Improvements
- Expand evaluation metrics (NDCG, coverage, diversity)
- Incorporate additional item metadata into content-based filtering (cast, director, year)
- Add automated bias/fairness analysis across user and item groups
- Deploy the backend and frontend to a public hosting service for a live demo link

## Author
Noor Fatima — Devnexes AI/ML Internship