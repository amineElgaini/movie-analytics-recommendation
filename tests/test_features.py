import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.features import runtime_category, clean_text, add_count_features, add_date_features


def test_runtime_category_short():
    assert runtime_category(80) == "short"


def test_runtime_category_medium():
    assert runtime_category(120) == "medium"


def test_runtime_category_long():
    assert runtime_category(200) == "long"


def test_runtime_category_unknown_for_nan():
    import numpy as np
    assert runtime_category(np.nan) == "unknown"


def test_clean_text_removes_punctuation_and_lowercases():
    result = clean_text("A Hero's Journey! In 2020.")
    assert result == "a hero s journey in"


def test_add_count_features_counts_list_lengths():
    df = pd.DataFrame({
        "genres": [["Action", "Drama"], ["Comedy"]],
        "keywords": [["a", "b", "c"], []],
    })
    result = add_count_features(df)
    assert result["n_genres"].tolist() == [2, 1]
    assert result["n_keywords"].tolist() == [3, 0]


def test_add_date_features_extracts_decade():
    df = pd.DataFrame({"release_date": pd.to_datetime(["2015-06-12", "1988-04-16"])})
    result = add_date_features(df)
    assert result["decade"].tolist() == [2010, 1980]