import pandas as pd
from src.preprocessing.cleaning import clean_main
from src.preprocessing.filling import fill_main


def setup(copy_df) -> pd.DataFrame:
    df = copy_df.copy()
    df.columns = [x.upper().strip() for x in df.columns]
    try:
        df.rename(columns=
                  {'FILM': 'TITLE',
                   'YEAR': 'RELEASE_YEAR',
                   'SCRIPT TYPE': 'SCRIPT_TYPE',
                   'ROTTEN TOMATOES  CRITICS': 'RT_CRITICS',
                   'METACRITIC  CRITICS': 'MC_CRITICS',
                   'ROTTEN TOMATOES AUDIENCE': 'RT_AUDIENCE',
                   'METACRITIC AUDIENCE': 'MC_AUDIENCE',
                   'PRIMARY GENRE': 'PRIMARY_GENRE',
                   'OPENING WEEKEND': 'OPENING_WEEKEND',
                   'DOMESTIC GROSS': 'DOMESTIC_GROSS',
                   'FOREIGN GROSS': 'FOREIGN_GROSS',
                   'WORLDWIDE GROSS': 'WORLDWIDE_GROSS',
                   'BUDGET ($MILLION)': 'BUDGET',
                   'IMDB RATING': 'IMDB_RATING',
                   'IMDB VS RT DISPARITY': 'IMDB_RT_DIFFERENCE',
                   'OSCAR WINNERS': 'WON_OSCAR',
                   'OSCAR DETAIL': 'OSCAR_DETAILS'},
                  inplace=True)
    except Exception as e:
        print(f"Problem in setup rename :{e}")
    try:
        df.drop(columns=
                ['DOMESTIC GROSS ($MILLION)',
                 'FOREIGN GROSS ($MILLION)',
                 'OPENING WEEKEND ($MILLION)',
                 'DISTRIBUTOR',
                 'WORLDWIDE GROSS ($MILLION)',
                 'OF GROSS EARNED ABROAD',
                 'ROTTEN TOMATOES VS METACRITIC  DEVIANCE',
                 'AVERAGE AUDIENCE',
                 'AVERAGE CRITICS',
                 'ROTTEN TOMATOES VS METACRITIC  DEVIANCE',
                 'BUDGET RECOVERED OPENING WEEKEND',
                 'BUDGET RECOVERED',
                 'AUDIENCE VS CRITICS DEVIANCE'], inplace=True)
    except Exception as e:
        print(f"Problem in setup drop :{e}")

    df = clean_main(df)
    df = fill_main(df)
    return df