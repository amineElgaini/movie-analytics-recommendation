import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Classification", layout="wide")
st.title("🎯 High Engagement Classifier")
st.markdown("Predict whether a movie would be in the top 25% most-voted (high engagement).")

@st.cache_resource
def load_model():
    return joblib.load("models/classifier_rf.pkl")

@st.cache_data
def load_data():
    return pd.read_pickle("data/processed/movies_final.pkl")

model = load_model()
df = load_data()

st.subheader("Enter movie details")

col1, col2 = st.columns(2)
with col1:
    runtime = st.slider("Runtime (minutes)", 40, 240, 110)
    budget = st.number_input("Budget ($)", 0, 500_000_000, 50_000_000, step=1_000_000)
    revenue = st.number_input("Revenue ($)", 0, 3_000_000_000, 100_000_000, step=1_000_000)
    popularity = st.slider("Popularity", 0.0, 100.0, 20.0)

with col2:
    n_genres = st.slider("Number of genres", 1, 6, 2)
    n_keywords = st.slider("Number of keywords", 0, 40, 10)
    overview_length = st.slider("Overview length (words)", 5, 150, 40)
    language = st.selectbox("Original language", sorted(df["original_language"].unique()))
    runtime_category = st.selectbox("Runtime category", ["short", "medium", "long"])

genre_options = sorted([c.replace("genre_", "") for c in df.columns if c.startswith("genre_")])
selected_genres = st.multiselect("Genres", genre_options)

if st.button("Predict"):
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    genre_values = {c: (1 if c.replace("genre_", "") in selected_genres else 0) for c in genre_cols}

    input_df = pd.DataFrame([{
        "runtime": runtime, "budget": budget, "revenue": revenue, "popularity": popularity,
        "n_genres": n_genres, "n_keywords": n_keywords, "overview_length": overview_length,
        "original_language": language, "runtime_category": runtime_category,
        **genre_values,
    }])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ High engagement predicted (confidence: {proba:.1%})")
    else:
        st.warning(f"⚠️ Not high engagement (confidence: {1-proba:.1%})")