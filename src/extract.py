import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
RAW_DIR = Path("data/raw")


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_json(endpoint, params=None, retries=3):
    """One safe API call. Returns a dict, or None if it fails."""
    params = {**(params or {}), "api_key": API_KEY}
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
            if r.status_code == 200:
                return r.json() or None          # empty answer -> None
            if r.status_code == 429:             # too many requests
                time.sleep(int(r.headers.get("Retry-After", 2)))
                continue
            if r.status_code == 404:             # movie not found
                return None
            print(f"HTTP {r.status_code} on {endpoint}")
        except requests.RequestException as e:   # network problem, timeout
            print(f"Network error on {endpoint}: {e}")
        time.sleep(2 ** attempt)                 # wait 1s, 2s, 4s
    return None


def fetch_movie_ids(n_pages=100):
    ids = []
    for page in range(1, n_pages + 1):
        path = RAW_DIR / "discover" / f"page_{page:03d}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = get_json("/discover/movie", {
                "page": page,
                "sort_by": "popularity.desc",
                "vote_count.gte": 20,
            })
            if not data or not data.get("results"):
                print(f"Stop at page {page}: no results")
                break
            save_json(data, path)
            time.sleep(0.1)
        ids += [m["id"] for m in data["results"]]
    return list(dict.fromkeys(ids))              # remove duplicates, keep order


def fetch_movie_details(ids):
    for i, movie_id in enumerate(ids, 1):
        path = RAW_DIR / "movies" / f"{movie_id}.json"
        if path.exists():
            continue
        data = get_json(f"/movie/{movie_id}", {"append_to_response": "keywords"})
        if data:
            save_json(data, path)
        time.sleep(0.1)
        if i % 100 == 0:
            print(f"{i}/{len(ids)} done")


if __name__ == "__main__":
    movie_ids = fetch_movie_ids(n_pages=100)
    print(f"{len(movie_ids)} movie ids found")
    fetch_movie_details(movie_ids)
    print("Extraction finished")