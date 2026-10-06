import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Clusters", layout="wide")
st.title("🧩 Movie Clusters")

@st.cache_data
def load_data():
    return pd.read_pickle("data/processed/movies_final.pkl")

df = load_data()

cluster_names = {
    0: "Mainstream mid-budget",
    1: "Acclaimed lower-budget",
    2: "Big-budget multi-genre",
    3: "Viral mega-hits (outliers)",
    4: "Premium tentpole blockbusters",
}
df["cluster_name"] = df["cluster"].map(cluster_names)

st.subheader("Cluster profiles")
cluster_features = ["runtime", "budget", "revenue", "popularity", "vote_average", "n_genres", "n_keywords"]
summary = df.groupby("cluster_name")[cluster_features].mean().round(2)
summary["count"] = df["cluster_name"].value_counts()
st.dataframe(summary, use_container_width=True)

fig = px.bar(summary.reset_index(), x="cluster_name", y="count", title="Movies per cluster")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Browse a cluster")
selected = st.selectbox("Choose a cluster", list(cluster_names.values()))
cluster_movies = df[df["cluster_name"] == selected][["title", "year", "vote_average", "popularity"]]
st.dataframe(cluster_movies.sort_values("popularity", ascending=False), use_container_width=True)