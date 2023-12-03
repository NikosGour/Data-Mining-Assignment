import os
import json
import pandas as pd
import requests
import Constants


def oscars_dataframe():
    # data taken from https://github.com/AminFadaee/Academy-Awards-Data
    data_folder = os.path.join(Constants.ROOT_DIR, 'src', 'oscar_data_preprocessing', 'winners_only')
    files = os.listdir(data_folder)
    files.sort()
    df = pd.DataFrame(columns=['year', 'winners', 'category'])
    for file_name in files:
        path = os.path.join(data_folder, file_name)
        file_name_without_extension = file_name.split('.')[0]
        with open(path) as f:
            data = json.load(f)
            for obj in data:
                if len(obj['winners']) == 2:
                    df.loc[len(df)] = [file_name_without_extension, obj['winners'][0], obj['category']]
                    df.loc[len(df)] = [file_name_without_extension, obj['winners'][1], obj['category']]
                else:
                    df.loc[len(df)] = [file_name_without_extension, obj['winners'][0], obj['category']]

    df.to_csv(os.path.join(Constants.ROOT_DIR, 'data', 'oscars.csv'), index=False)


def get_imdb_rating(df: pd.DataFrame):
    responses = []
    data_folder = os.path.join(Constants.ROOT_DIR, 'src', 'oscar_data_preprocessing', 'imdb_ratings')

    with open(os.path.join(data_folder, 'imdb_ratings.json'), "w") as f:
        for film in df['Film']:
            url = f"https://www.omdbapi.com/?t={film}&apikey=6e69d163"
            response = requests.get(url)
            try:
                json_res = response.json()
                responses.append(json_res)
                print(json_res)
            except:
                print(response.text)
        json.dump(responses, f)

def imdb_json_to_csv():
    res_columns = ["Title", "Year", "Rated", "Released", "Runtime", "Genre", "Director", "Writer", "Actors", "Plot",
                   "Language", "Country", "Awards", "Poster", "Ratings", "Metascore", "imdbRating", "imdbVotes",
                   "imdbID", "Type", "DVD", "BoxOffice", "Production", "Website", "Response"]

    data_folder = os.path.join(Constants.ROOT_DIR, 'src', 'oscar_data_preprocessing', 'imdb_ratings')
    with open(os.path.join(data_folder, 'imdb_ratings.json')) as f:
        data = json.load(f)
        df = pd.DataFrame(data,columns=res_columns)
        df.to_csv(os.path.join(Constants.ROOT_DIR, 'data', 'imdb_ratings.csv'), index=False)