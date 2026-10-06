import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def load_recommendation_data():
    df = pd.read_pickle("data/processed/movies_clustered.pkl")
    tfidf_matrix = sparse.load_npz("models/tfidf_matrix.npz")
    return df, tfidf_matrix


def recommend(title, df, tfidf_matrix, n=5):
    matches = df.index[df["title"].str.lower() == title.lower()]
    if len(matches) == 0:
        return None

    idx = matches[0]
    movie_vector = tfidf_matrix[idx]
    sims = cosine_similarity(movie_vector, tfidf_matrix).flatten()

    similar_idx = sims.argsort()[::-1]
    similar_idx = [i for i in similar_idx if i != idx][:n]

    results = df.iloc[similar_idx][["title", "genres", "year"]].copy()
    results["similarity"] = sims[similar_idx].round(3)
    return results.reset_index(drop=True)
