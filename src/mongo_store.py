import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")


def get_collection():
    client = MongoClient(MONGO_URI)
    return client["movie_db"]["movies"]


def df_to_documents(df):
    # MongoDB does not understand NaN, so we use None (null)
    df = df.astype(object).where(df.notna(), None)
    docs = df.to_dict("records")
    for d in docs:
        d["_id"] = d["movie_id"]                              # one document per movie
        d["release_date"] = d["release_date"].to_pydatetime()  # Pandas date -> Python date
    return docs


def store_movies(df):
    col = get_collection()
    ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in df_to_documents(df)]
    result = col.bulk_write(ops)
    col.create_index("vote_average")
    col.create_index("genres")
    print("inserted:", result.upserted_count, "| updated:", result.modified_count)
    print("total in collection:", col.count_documents({}))


if __name__ == "__main__":
    df = pd.read_pickle("data/processed/movies_clean.pkl")
    store_movies(df)