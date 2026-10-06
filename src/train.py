import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = ["runtime", "budget", "revenue", "popularity", "n_genres", "n_keywords", "overview_length"]
CATEGORICAL_FEATURES = ["original_language", "runtime_category"]


def build_tfidf(df):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["overview_clean"])
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    sparse.save_npz(MODELS_DIR / "tfidf_matrix.npz", tfidf_matrix)
    return tfidf_matrix


def add_genre_columns(df):
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df["genres"])
    genre_cols = [f"genre_{g}" for g in mlb.classes_]
    genre_df = pd.DataFrame(genre_matrix, columns=genre_cols, index=df.index)
    return pd.concat([df, genre_df], axis=1), genre_cols


def train_classifier(df, genre_cols):
    threshold = df["vote_count"].quantile(0.75)
    df["high_engagement"] = (df["vote_count"] >= threshold).astype(int)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + genre_cols]
    y = df["high_engagement"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ("genre", "passthrough", genre_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4],
    }
    grid = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    joblib.dump(grid.best_estimator_, MODELS_DIR / "classifier_rf.pkl")
    return df


def train_clustering(df):
    cluster_features = ["runtime", "budget", "revenue", "popularity", "vote_average", "n_genres", "n_keywords"]
    cluster_df = df[cluster_features].fillna(df[cluster_features].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.pkl")
    return df


if __name__ == "__main__":
    df = pd.read_pickle("data/processed/movies_features.pkl")
    build_tfidf(df)
    df, genre_cols = add_genre_columns(df)
    df = train_classifier(df, genre_cols)
    df = train_clustering(df)
    df.to_pickle("data/processed/movies_final.pkl")
    print("Training complete:", df.shape)