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
    res_columns = ["Title", "Year", "Rated", "Released", "Runtime", "Genre", "Director", "Writer", "Actors", "Plot",
                   "Language", "Country", "Awards", "Poster", "Ratings", "Metascore", "imdbRating", "imdbVotes",
                   "imdbID", "Type", "DVD", "BoxOffice", "Production", "Website", "Response"]

    responses = []
    df_prev = pd.read_csv(os.path.join(Constants.ROOT_DIR, 'data', 'imdb_ratings.csv'))


    for film in df['Film']:
        if film in df_prev['Title'].values:
            print(f'{film} already in csv')
            responses.append(df_prev[df_prev['Title'] == film].iloc[0])
            continue
        url = f"https://www.omdbapi.com/?t={film}&apikey=6e69d163"
        response = requests.get(url)
        try:
            json_res = response.json()
            responses.append(json_res)
            print(json_res)
        except:
            print(response.text)
    df_res = pd.DataFrame(responses, columns=res_columns)
    df_res.to_csv(os.path.join(Constants.ROOT_DIR, 'data', 'imdb_ratings.csv'), index=False)


def quick_fix(df):
    df_cur = pd.read_csv(os.path.join(Constants.ROOT_DIR, 'data', 'imdb_ratings.csv'))
    df_copy = pd.read_csv(os.path.join(Constants.ROOT_DIR, 'data', 'imdb_ratings_copy.csv'))

    movies = []
    movies_not_found = []


    for film in df['Film']:
        film = str(film)
        if film in df_cur['Title'].values and film in df_copy['Title'].values:
            movies.append(df_copy[df_copy['Title'] == film].iloc[0])
            print(f"<<{film}>> in both")
        elif film in df_cur['Title'].values:
            movies.append(df_cur[df_cur['Title'] == film].iloc[0])
            print(f"<<{film}>> in cur")
        elif film in df_copy['Title'].values:
            movies.append(df_copy[df_copy['Title'] == film].iloc[0])
            print(f"<<{film}>> in copy")
        else:
            movies_not_found.append(film)
            print(f"<<{film}>> not found")

    df_res = pd.DataFrame(movies, columns=df_cur.columns)
    df_res.to_csv(os.path.join(Constants.ROOT_DIR, 'data', 'imdb_ratings_new.csv'), index=False)
    df_not_found = pd.DataFrame(movies_not_found, columns=['Film'])
    df_not_found.to_csv(os.path.join(Constants.ROOT_DIR, 'data', 'not_found.csv'), index=False)

