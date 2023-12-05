import json
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import requests
import pandas as pd
import Constants
import os
import src.preprocessing.preprocessing as preprocessing
from IPython.display import display, HTML



df = pd.read_excel('data/movies.xlsx')
df = preprocessing.setup(df)
print(df['WON_OSCAR'])


# movie_name = 'WALL-E'
# url = f"https://www.omdbapi.com/?t={movie_name}&apikey=10461461"
#
#
# response = requests.get(url)
#
# print(response.json()['imdbRating'])

# df1 = pd.read_csv('data/imdb_ratings.csv')
# df2 = pd.read_csv('data/not_found.csv')
#
# print(len(df1)+len(df2) == len(df))

