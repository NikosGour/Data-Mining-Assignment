import json
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import requests
import pandas as pd
import Constants
import os

# print(df)
# movie_name = 'WALL-E'
# url = f"https://www.omdbapi.com/?t={movie_name}&apikey=10461461"
#
#
# response = requests.get(url)
#
# print(response.json()['imdbRating'])

# df = pd.read_excel('data/movies.xlsx')
# df1 = pd.read_csv('data/imdb_ratings.csv')
# df2 = pd.read_csv('data/not_found.csv')
#
# print(len(df1)+len(df2) == len(df))