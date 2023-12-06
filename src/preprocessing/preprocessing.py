import pandas as pd
from src.preprocessing.cleaning import clean_column_values
from src.preprocessing.filling import fill_column_values

def fix_column_names(df: pd.DataFrame):
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
        # We drop the columns that we don't need, or that we will calculate later see note at the end of the file.
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
                 'AUDIENCE VS CRITICS DEVIANCE',
                 'PRIMARY_GENRE'], inplace=True)
    except Exception as e:
        print(f"Problem in setup drop :{e}")
    return df

def additional_preprocessing(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=["GENRE", 'SCRIPT_TYPE', 'OSCAR_DETAILS'])
    # df = df.drop(columns=['TITLE'])
    return df
def setup(copy_df) -> pd.DataFrame:
    df = copy_df.copy()

    df = fix_column_names(df)
    df = clean_column_values(df)
    df = fill_column_values(df)
    df = additional_preprocessing(df)
    return df


'''
NOTE : drop (above line need to be recalculated for better accuracy, see comment bellow for how I recalculate. Bellow line just get dropped)
        ✅`of Gross earned abroad`
        ✅`Budget ($million)`
        ✅`Budget recovered`
        ✅`Budget recovered opening weekend`
        ✅`Average critics`
        ✅`Rotten Tomatoes vs Metacritic  deviance`
        ✅`Average audience`
        ✅`Audience vs Critics deviance`
        ✅`Release Date (US)`
        --- --- --- --- --- ---
        ✅`Domestic gross ($million)` : have more accurate data in `DOMESTIC_GROSS`
        ✅`Foreign Gross ($million)` : have more accurate data in `FOREIGN_GROSS`
        ✅`Opening weekend ($million)` : have more accurate data in `OPENING_WEEKEND`
        ✅`Distributor` : have 0 data
        ✅`Worldwide Gross ($million)` : have more accurate data in `WORLDWIDE_GROSS`
        ✅`PRIMARY_GENRE` : have way too little data
'''

'''
NOTE : Calculate bellow (Field bellow line are new)
    ✅ `of Gross earned abroad`: (`Foreign Gross`/ `Worldwide Gross`) * 100
    ✅ `Budget ($million)`: `Budget ($million)` * 1_000_000
    ✅ `Budget recovered`: (`Worldwide Gross` / `Budget ($million)`) * 100
    ✅ `Budget recovered opening weekend`: (`Opening Weekend` / `Budget ($million)`) * 100
    ✅ `Average critics`: (`Rotten Tomatoes  critics` + `Metacritic  critics`)/2
    ✅ `Rotten Tomatoes vs Metacritic  deviance`: `Rotten Tomatoes Audience` - `Metacritic Audience`
    ✅ `Average audience`: (`Rotten Tomatoes Audience` + `Metacritic Audience`)/2
    ✅ `Audience vs Critics deviance`: `Average critics` - `Average audience`
    --- --- --- --- --- --- ---
    ✅ `Month`: parse the month out of `Release Date (US)`
    ✅ `Day`: parse the day out of `Release Date (US)`

'''