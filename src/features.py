import re
import pandas as pd
import numpy as np


def add_date_features(df):
    df["year"] = df["release_date"].dt.year
    df["month"] = df["release_date"].dt.month
    df["decade"] = (df["year"] // 10) * 10
    return df


def add_count_features(df):
    df["n_genres"] = df["genres"].apply(len)
    df["n_keywords"] = df["keywords"].apply(len)
    return df


def runtime_category(minutes):
    if pd.isna(minutes):
        return "unknown"
    elif minutes < 90:
        return "short"
    elif minutes <= 150:
        return "medium"
    return "long"


def add_runtime_category(df):
    df["runtime_category"] = df["runtime"].apply(runtime_category)
    return df


def add_overview_length(df):
    df["overview_length"] = df["overview"].apply(lambda x: len(x.split()))
    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_clean_overview(df):
    df["overview_clean"] = df["overview"].apply(clean_text)
    return df


def build_features(df):
    df = add_date_features(df)
    df = add_count_features(df)
    df = add_runtime_category(df)
    df = add_overview_length(df)
    df = add_clean_overview(df)
    return df


if __name__ == "__main__":
    df = pd.read_pickle("data/processed/movies_clean.pkl")
    df = build_features(df)
    df.to_pickle("data/processed/movies_features.pkl")
    print("Features saved:", df.shape)