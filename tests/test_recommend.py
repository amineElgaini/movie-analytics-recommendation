import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy import sparse
from src.recommend import recommend


def make_fake_data():
    df = pd.DataFrame({
        "title": ["Movie A", "Movie B", "Movie C"],
        "genres": [["Action"], ["Drama"], ["Action"]],
        "year": [2020, 2021, 2022],
    })
    # Fake TF-IDF: Movie A and C share word patterns, Movie B is different
    tfidf_matrix = sparse.csr_matrix([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
    ])
    return df, tfidf_matrix


def test_recommend_excludes_the_queried_movie_itself():
    df, tfidf_matrix = make_fake_data()
    results = recommend("Movie A", df, tfidf_matrix, n=2)
    assert "Movie A" not in results["title"].tolist()


def test_recommend_returns_most_similar_first():
    df, tfidf_matrix = make_fake_data()
    results = recommend("Movie A", df, tfidf_matrix, n=2)
    assert results.iloc[0]["title"] == "Movie C"


def test_recommend_returns_none_for_unknown_title():
    df, tfidf_matrix = make_fake_data()
    result = recommend("Nonexistent Movie", df, tfidf_matrix, n=2)
    assert result is None


def test_recommend_is_case_insensitive():
    df, tfidf_matrix = make_fake_data()
    results = recommend("movie a", df, tfidf_matrix, n=2)
    assert results is not None