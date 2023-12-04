import json

import requests



movie_name = 'WALL-E'
url = f"https://www.omdbapi.com/?t={movie_name}&apikey=10461461"


response = requests.get(url)

print(response.json()['imdbRating'])