import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from src.clean import clean


def test_clean_converts_zero_budget_to_nan():
    df = pd.DataFrame({
        "movie_id": [1], "title": ["Test"], "overview": ["A test movie"],
        "release_date": ["2020-01-01"], "runtime": [100],
        "original_language": ["en"], "genres": [["Drama"]], "keywords": [["test"]],
        "budget": [0], "revenue": [1000], "popularity": [10.0],
        "vote_average": [7.0], "vote_count": [100],
    })
    result = clean(df)
    assert pd.isna(result.loc[0, "budget"])


def test_clean_keeps_real_zero_as_is_for_revenue_not_budget():
    df = pd.DataFrame({
        "movie_id": [1], "title": ["Test"], "overview": ["A test movie"],
        "release_date": ["2020-01-01"], "runtime": [100],
        "original_language": ["en"], "genres": [["Drama"]], "keywords": [["test"]],
        "budget": [5000000], "revenue": [0], "popularity": [10.0],
        "vote_average": [7.0], "vote_count": [100],
    })
    result = clean(df)
    assert pd.isna(result.loc[0, "revenue"])


def test_clean_drops_rows_with_empty_overview():
    df = pd.DataFrame({
        "movie_id": [1, 2], "title": ["A", "B"], "overview": ["", "Real overview here"],
        "release_date": ["2020-01-01", "2020-01-01"], "runtime": [100, 100],
        "original_language": ["en", "en"], "genres": [["Drama"], ["Drama"]],
        "keywords": [["test"], ["test"]], "budget": [1, 1], "revenue": [1, 1],
        "popularity": [1.0, 1.0], "vote_average": [1.0, 1.0], "vote_count": [1, 1],
    })
    result = clean(df)
    assert len(result) == 1
    assert result.iloc[0]["title"] == "B"