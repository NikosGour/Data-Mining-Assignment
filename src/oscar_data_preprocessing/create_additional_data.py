import os
import json
import pandas as pd
import requests
import Constants
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()

def create_oscars_csv():
    # data taken from https://github.com/AminFadaee/Academy-Awards-Data
    data_folder = os.path.join(Constants.ROOT_PROJECT_DIR, 'src', 'oscar_data_preprocessing', 'winners_only')
    json_files = os.listdir(data_folder)
    json_files.sort()

    df = pd.DataFrame(columns=['year', 'winners', 'category'])

    for file_name in json_files:

        path = os.path.join(data_folder, file_name)
        file_name_without_extension = file_name.split('.')[0]

        with open(path) as f:
            data = json.load(f)

            for obj in data:

                #After analysis of the data, there is only two cases,for each oscar category there can be:
                #1. Two winners
                #2. One winner
                #That's why I'm using this if statement instead of a more generalized approach
                if len(obj['winners']) == 2:

                    #Appending a new row to the `Oscars dataframe` with columns:
                    # year: the year of the oscar
                    # winners: the name of the Movie that won
                    # category: the Title of the oscar it won
                    df.loc[len(df)] = [file_name_without_extension, obj['winners'][0], obj['category']]
                    df.loc[len(df)] = [file_name_without_extension, obj['winners'][1], obj['category']]
                else:
                    df.loc[len(df)] = [file_name_without_extension, obj['winners'][0], obj['category']]

    df.to_csv(os.path.join(Constants.ROOT_PROJECT_DIR, 'data', 'oscars.csv'), index=False)


def create_enhanced_movie_data_csv(df: pd.DataFrame):
    # Data taken from https://www.omdbapi.com/
    # Res_columns are the columns that are returned from the API and are used to create the csv
    res_columns = ["Title", "Year", "Rated", "Released", "Runtime", "Genre", "Director", "Writer", "Actors", "Plot",
                   "Language", "Country", "Awards", "Poster", "Ratings", "Metascore", "imdbRating", "imdbVotes",
                   "imdbID", "Type", "DVD", "BoxOffice", "Production", "Website", "Response"]

    responses = []
    movies_not_found = []

    # df_prev is used incase the script is stopped in the middle of the process, and we want to continue
    # from where we left off, or if the api failed on some movies, and we want to retry them later
    output_file_path = os.path.join(Constants.ROOT_PROJECT_DIR, 'data', 'imdb_ratings.csv')

    if os.path.exists(output_file_path):
        df_prev = pd.read_csv(output_file_path)
    else:
        df_prev = pd.DataFrame(columns=res_columns)

    for film in df['Film']:

        # This is needed because if we have a movie with a name that can be interpreted as a number,
        # then when checking if the movie is in the previous dataframe, it will be interpreted as a number and
        # will not be found, so we convert it to a string

        # Example: Movie name is '300'
        # df_prev['Title'].values = ['300', 'The Godfather']
        # when the `for` statement is run above the field `film` will be 300 with type `int`, and in the `df_prev`
        # the field `Title` is of type `str`, so the comparison will fail
        film = str(film)

        if film in df_prev['Title'].values:

            # `Fore.Cyan` and `Style.RESET_ALL` are used to color the text in the terminal for better readability
            print(f'{Fore.CYAN}{film} already in csv{Style.RESET_ALL}')
            responses.append(df_prev[df_prev['Title'] == film].iloc[0])
            continue

        url = f"https://www.omdbapi.com/?t={film}&apikey=6e69d163"
        response = requests.get(url)

        try:
            json_res = response.json()

            # The API returns `False` in the `Response` field if the movie was not found
            if json_res['Response'] == 'False':
                print(f'{Fore.RED}Error in {film}, with url: {url} and response: {response.text}{Style.RESET_ALL}')
                movies_not_found.append(film)
            else:
                print(f"Found Movie: {json_res}")

                # I found a bug in Pandas (and issued it: https://github.com/pandas-dev/pandas/issues/56322) and
                # also found a workaround for it, that's why `pd.Series()` is there. Read the issue for more info
                responses.append(pd.Series(json_res))

        except json.decoder.JSONDecodeError:
            print(f'{Fore.RED}Error in {film}, with url: {url} and response: {response.text}{Style.RESET_ALL}')


    df_res = pd.DataFrame(responses, columns=res_columns)
    df_res.to_csv(output_file_path, index=False)

    df_not_found = pd.DataFrame(movies_not_found, columns=['Film'])
    df_not_found.to_csv(os.path.join(Constants.ROOT_PROJECT_DIR, 'data', 'not_found.csv'), index=False)
