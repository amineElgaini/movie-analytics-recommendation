import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from src.recommend import load_recommendation_data, recommend

st.set_page_config(page_title="Recommendation", layout="wide")
st.title("🎥 Movie Recommendation")

df, tfidf_matrix = load_recommendation_data()

movie_title = st.selectbox("Choose a movie", sorted(df["title"].unique()))

if st.button("Get recommendations"):
    results = recommend(movie_title, df, tfidf_matrix)
    if results is None:
        st.error("Movie not found.")
    else:
        st.subheader(f"Movies similar to '{movie_title}'")
        st.dataframe(results, use_container_width=True)