import json
from pathlib import Path

import pandas as pd
import numpy as np

def load_raw_movies(raw_dir="data/raw/movies"):
    rows = []
    for path in Path(raw_dir).glob("*.json"):
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "movie_id": m.get("id"),
            "title": m.get("title"),
            "overview": m.get("overview"),
            "release_date": m.get("release_date"),
            "runtime": m.get("runtime"),
            "original_language": m.get("original_language"),
            "genres": [g["name"] for g in m.get("genres", [])],
            "keywords": [k["name"] for k in m.get("keywords", {}).get("keywords", [])],
            "budget": m.get("budget"),
            "revenue": m.get("revenue"),
            "popularity": m.get("popularity"),
            "vote_average": m.get("vote_average"),
            "vote_count": m.get("vote_count"),
            "poster_path": m.get("poster_path")
        })
    return pd.DataFrame(rows)


def inspect(df):
    print("Shape:", df.shape)
    print("\n--- Types ---")
    print(df.dtypes)
    print("\n--- Missing values ---")
    print(df.isna().sum())
    print("\n--- Empty text ---")
    print("empty overview:", (df["overview"].fillna("").str.strip() == "").sum())
    print("\n--- Duplicates ---")
    print("duplicate movie_id:", df.duplicated("movie_id").sum())
    print("duplicate title+date:", df.duplicated(["title", "release_date"]).sum())
    print("\n--- Suspicious zeros ---")
    for col in ["budget", "revenue", "runtime"]:
        print(f"{col} == 0:", (df[col] == 0).sum())
    print("\n--- Numeric summary ---")
    print(df[["runtime", "budget", "revenue", "popularity", "vote_average", "vote_count"]].describe())


def clean(df):
    df = df.copy()
    df = df.drop_duplicates("movie_id")

    # 0 means "unknown" in TMDB
    for col in ["budget", "revenue", "runtime"]:
        df[col] = df[col].replace(0, np.nan)

    # dates
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # text
    df["title"] = df["title"].str.strip()
    df["overview"] = df["overview"].fillna("").str.strip()

    # remove rows we cannot use
    df = df.dropna(subset=["release_date"])
    df = df[df["overview"] != ""]
    return df.reset_index(drop=True)


NUMERIC_COLS = ["runtime", "budget", "revenue", "popularity", "vote_average", "vote_count"]
CATEGORICAL_COLS = ["original_language", "genres"]
TEXT_COLS = ["title", "overview", "keywords"]


if __name__ == "__main__":
    df = load_raw_movies()
    # inspect(df)
    clean_df = clean(df)

    print("Before:", df.shape, "-> After:", clean_df.shape)
    print("\nMissing values now:")
    print(clean_df[NUMERIC_COLS].isna().sum())
    print("\nFuture release dates:", (clean_df["release_date"] > pd.Timestamp.today()).sum())
    print("Revenue > 0 but budget unknown:", (clean_df["revenue"].notna() & clean_df["budget"].isna()).sum())
    print("\nDates from", clean_df["release_date"].min().date(), "to", clean_df["release_date"].max().date())

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    clean_df.to_pickle("data/processed/movies_clean.pkl")
    print("\nSaved data/processed/movies_clean.pkl")