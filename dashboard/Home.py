import streamlit as st
import pandas as pd

st.set_page_config(page_title="Movie Intelligence", layout="wide")

st.title("🎬 Movie Intelligence")
st.markdown("Analyse, classification, clustering et recommandation de films à partir de données TMDB.")

df = pd.read_pickle("data/processed/movies_clustered.pkl")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Movies", len(df))
col2.metric("Avg rating", round(df["vote_average"].mean(), 2))
col3.metric("Date range", f"{df['year'].min()}–{df['year'].max()}")
col4.metric("Clusters", df["cluster"].nunique())

st.markdown("Use the sidebar to explore the dashboard, classification, clusters, and recommendations.")