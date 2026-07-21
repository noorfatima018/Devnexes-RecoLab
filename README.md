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
- [ ] Content-based filtering (in progress)
- [ ] Hybrid model with cold-start handling (in progress)
- [ ] Interactive interface (Streamlit)

## Technology Stack
- **Language:** Python 3.14
- **Data Processing:** pandas, numpy
- **ML/Recommendation:** scikit-learn (scikit-surprise attempted but blocked — see Limitations)
- **Visualization:** matplotlib, seaborn
- **Interface (planned):** Streamlit
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

7. Run full evaluation (compares all models):
```bash
python -m src.evaluation
```

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

**Dataset Insights:**
- Sparsity: ~93.7% of user-movie combinations have no rating
- Long-tail distribution: majority of movies have very few ratings — highlighting the
  cold-start challenge this project aims to solve

## Testing Notes
- Manual verification of data loading against known MovieLens statistics (943 users, 1682 movies, 100000 ratings) — confirmed match
- Baseline model output manually reviewed for sanity (recommends well-known, highly-rated films)
- Collaborative filtering training RMSE monitored across epochs to confirm stable convergence
- Manually verified that different users receive different recommendation lists (confirms personalization is working)

## Limitations (Current Stage)
- `scikit-surprise` library could not be used due to a system-level Application Control
  policy blocking its compiled binaries — collaborative filtering was implemented manually
  using NumPy as a result
- Cold-start handling not yet implemented (planned for upcoming weeks)
- Content-based filtering not yet implemented
- Evaluation currently limited to Precision@K; additional metrics (NDCG, Recall@K) planned
- Collaborative filtering currently uses random initialization with a fixed seed;
  hyperparameters (factors, learning rate, epochs) not yet tuned

## Future Improvements
- Add content-based filtering using movie genres
- Combine into hybrid model with cold-start fallback logic
- Build Streamlit interface for interactive testing
- Expand evaluation metrics (NDCG, coverage, diversity)
- Tune collaborative filtering hyperparameters (factors, learning rate, epochs, regularization)

## Author
Noor Fatima — Devnexes AI/ML Internship