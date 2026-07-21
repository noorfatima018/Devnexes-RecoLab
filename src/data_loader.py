import pandas as pd
from sklearn.model_selection import train_test_split


def load_ratings(path='data/raw/ml-100k/u.data'):
    """User ratings load karta hai: user_id, item_id, rating, timestamp"""
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(path, sep='\t', names=cols)
    return ratings


def load_movies(path='data/raw/ml-100k/u.item'):
    """Movie metadata load karta hai: title, genres, etc."""
    cols = ['item_id', 'title', 'release_date', 'video_release_date',
             'imdb_url'] + [f'genre_{i}' for i in range(19)]
    movies = pd.read_csv(path, sep='|', names=cols, encoding='latin-1')
    return movies


def load_users(path='data/raw/ml-100k/u.user'):
    """User demographics load karta hai: age, gender, occupation, zip"""
    cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv(path, sep='|', names=cols)
    return users


def get_train_test_split(ratings, test_size=0.2, random_state=42):
    """
    Ratings ko train aur test mein split karta hai.
    Random split use kar rahe hain (stratify nahi kar rahe kyunke
    har user/item ke paas kaafi ratings nahi hoti stratify ke liye).
    """
    train, test = train_test_split(
        ratings, test_size=test_size, random_state=random_state
    )
    return train, test


if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()

    print("Ratings sample:")
    print(ratings.head())
    print(f"\nTotal ratings: {len(ratings)}")

    print("\nMovies sample:")
    print(movies[['item_id', 'title']].head())

    print("\nUsers sample:")
    print(users.head())