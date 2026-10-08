import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from src.recommend import load_recommendation_data, recommend

st.set_page_config(page_title="Recommendation", layout="wide")
st.title("🎥 Movie Recommendation")

df, tfidf_matrix = load_recommendation_data()

movie_title = st.selectbox("Choose a movie", sorted(df["title"].unique()))

POSTER_BASE = "https://image.tmdb.org/t/p/w342"

if st.button("Get recommendations"):
    results = recommend(movie_title, df, tfidf_matrix)
    if results is None:
        st.error("Movie not found.")
    else:
        st.subheader(f"Movies similar to '{movie_title}'")
        cols = st.columns(5)
        for col, (_, row) in zip(cols, results.iterrows()):
            with col:
                if isinstance(row["poster_path"], str):
                    st.image(POSTER_BASE + row["poster_path"])
                else:
                    st.write("No poster")
                st.markdown(f"**{row['title']}** ({row['year']})")
                st.caption(", ".join(row["genres"]))
                st.caption(f"Similarity: {row['similarity']}")