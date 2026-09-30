from mongo_store import get_collection

col = get_collection()

# 1. Top 5 rated movies (with enough votes)
print("\n1. Best rated movies")
for m in col.find({"vote_count": {"$gte": 5000}},
                  {"_id": 0, "title": 1, "vote_average": 1}
                  ).sort("vote_average", -1).limit(5):
    print(m)

# 2. Count French movies
print("\n2. French movies:", col.count_documents({"original_language": "fr"}))

# 3. Short Drama movies (a value inside a list)
print("\n3. Drama movies under 100 min")
for m in col.find({"genres": "Drama", "runtime": {"$lt": 100}},
                  {"_id": 0, "title": 1, "runtime": 1}).limit(5):
    print(m)

# 4. Movies with both budget and revenue known
print("\n4. Movies with budget and revenue:",
      col.count_documents({"budget": {"$ne": None}, "revenue": {"$ne": None}}))

# 5. AGGREGATION: average rating per genre
pipeline = [
    {"$unwind": "$genres"},                       # one row per genre of each movie
    {"$group": {
        "_id": "$genres",
        "n_movies": {"$sum": 1},
        "avg_rating": {"$avg": "$vote_average"},
        "avg_votes": {"$avg": "$vote_count"},
    }},
    {"$match": {"n_movies": {"$gte": 20}}},       # ignore very small genres
    {"$sort": {"avg_rating": -1}},
    {"$project": {
        "_id": 0,
        "genre": "$_id",
        "n_movies": 1,
        "avg_rating": {"$round": ["$avg_rating", 2]},
        "avg_votes": {"$round": ["$avg_votes", 0]},
    }},
]
print("\n5. Aggregation: rating per genre")
for row in col.aggregate(pipeline):
    print(row)