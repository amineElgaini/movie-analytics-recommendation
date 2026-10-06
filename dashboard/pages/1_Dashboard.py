import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 Dashboard")

@st.cache_data
def load_data():
    return pd.read_pickle("data/processed/movies_final.pkl")

df = load_data()

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x="vote_average", nbins=30, title="Vote average distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(df, x="popularity", nbins=30, range_x=[0, 100], title="Popularity distribution")
    st.plotly_chart(fig, use_container_width=True)

genre_counts = df.explode("genres")["genres"].value_counts().reset_index()
genre_counts.columns = ["genre", "count"]
fig = px.bar(genre_counts, x="count", y="genre", orientation="h", title="Movies per genre")
st.plotly_chart(fig, use_container_width=True)

year_counts = df["year"].value_counts().sort_index().reset_index()
year_counts.columns = ["year", "count"]
fig = px.line(year_counts, x="year", y="count", title="Movies by release year")
st.plotly_chart(fig, use_container_width=True)