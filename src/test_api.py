import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads the .env file
api_key = os.getenv("TMDB_API_KEY")

url = "https://api.themoviedb.org/3/movie/550"
response = requests.get(url, params={"api_key": api_key}, timeout=10)

print(response.status_code)   # 200 means OK
print(response.json()["title"])  # Fight Club