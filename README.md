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
- [ ] Collaborative filtering (in progress — Week 2)
- [ ] Content-based filtering (in progress)
- [ ] Hybrid model with cold-start handling (in progress)
- [ ] Interactive interface (Streamlit)

## Technology Stack
- **Language:** Python 3.14
- **Data Processing:** pandas, numpy
- **ML/Recommendation:** scikit-learn, scikit-surprise
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

6. Run evaluation:
```bash
python -m src.evaluation
```

## Results So Far

**Baseline Model (Popularity-Based):**
- Average Precision@10: **0.0625**
- Top recommended movies: Star Wars (1977), Schindler's List, Shawshank Redemption, Godfather

**Dataset Insights:**
- Sparsity: ~93.7% of user-movie combinations have no rating
- Long-tail distribution: majority of movies have very few ratings — highlighting the
  cold-start challenge this project aims to solve

## Testing Notes
- Manual verification of data loading against known MovieLens statistics (943 users, 1682 movies, 100000 ratings) — confirmed match
- Baseline model output manually reviewed for sanity (recommends well-known, highly-rated films)

## Limitations (Current Stage)
- Baseline model is not personalized — same recommendations for every user
- Cold-start handling not yet implemented (planned for later weeks)
- Evaluation currently limited to Precision@K; additional metrics (NDCG, Recall@K) planned

## Future Improvements
- Add collaborative filtering (SVD-based matrix factorization)
- Add content-based filtering using movie genres
- Combine into hybrid model with cold-start fallback logic
- Build Streamlit interface for interactive testing
- Expand evaluation metrics (NDCG, coverage, diversity)

## Author
Noor Fatima — Devnexes AI/ML Internship