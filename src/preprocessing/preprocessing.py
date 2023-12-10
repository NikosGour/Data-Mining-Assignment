import pandas as pd

from src.predict_class import PredictionMovie
from src.preprocessing.cleaning import clean_column_values
from src.preprocessing.filling import fill_column_values

class Preprocessing():
    def __init__(self):
        self.columns = None
        self.train_df = None
    def fix_column_names(self,df: pd.DataFrame):
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


    def fit(self,copy_df) -> pd.DataFrame:
        df = copy_df.copy()


        df = self.fix_column_names(df)
        df = clean_column_values(df)
        df = fill_column_values(df)
        df.drop(columns=['OSCAR_DETAILS'],inplace=True)
        self.train_df = df
        self.columns = df.columns
        return df


    def transform(self, movies: list[PredictionMovie]) -> pd.DataFrame:

        df = pd.DataFrame(columns=self.columns)
        df = df.astype(self.train_df.dtypes.to_dict())

        for i,movie in enumerate(movies):
            df.loc[i, 'TITLE'] = movie.title
            df.loc[i, 'RELEASE_YEAR'] = movie.year
            df.loc[i, 'RT_CRITICS'] = movie.rt_critics
            df.loc[i, 'MC_CRITICS'] = movie.mc_critics
            df.loc[i, 'RT_AUDIENCE'] = movie.rt_audience
            df.loc[i, 'MC_AUDIENCE'] = movie.mc_audience
            df.loc[i, 'OPENING_WEEKEND'] = movie.opening_weekend
            df.loc[i, 'BUDGET'] = movie.budget * 1_000_000
            df.loc[i, 'WORLDWIDE_GROSS'] = movie.worldwide_gross
            df.loc[i, 'FOREIGN_GROSS'] = movie.foreign_gross
            df.loc[i, 'DOMESTIC_GROSS'] = movie.domestic_gross
            df.loc[i, 'IMDB_RATING'] = movie.imdb_rating * 10
            df.loc[i, 'IMDB_RT_DIFFERENCE'] = df.loc[i, 'IMDB_RATING'] - df.loc[i, 'RT_CRITICS']
            df.loc[i, 'WON_OSCAR'] = pd.NA
            df.loc[i, 'RELEASE_MONTH'] = movie.release_month
            df.loc[i, 'RELEASE_DAY'] = movie.release_day
            df.loc[i, 'RT_MC_AUDIENCE_DIFFERENCE'] = df.loc[i, 'RT_AUDIENCE'] - df.loc[i, 'MC_AUDIENCE']
            df.loc[i, 'AVERAGE_AUDIENCE'] = (df.loc[i, 'RT_AUDIENCE'] + df.loc[i, 'MC_AUDIENCE']) / 2
            df.loc[i, 'AVERAGE_CRITICS'] = (df.loc[i, 'RT_CRITICS'] + df.loc[i, 'MC_CRITICS']) / 2
            df.loc[i, 'CRITICS_AUDIENCE_DIFFERENCE'] = df.loc[i, 'AVERAGE_CRITICS'] - df.loc[i, 'AVERAGE_AUDIENCE']
            df.loc[i,'%BUDGET_RECOVERED'] = (df.loc[i, 'WORLDWIDE_GROSS'] / df.loc[i, 'BUDGET']) * 100
            df.loc[i,'%BUDGET_RECOVERED_OPENING_WEEKEND'] = (df.loc[i, 'OPENING_WEEKEND'] / df.loc[i, 'BUDGET']) * 100
            df.loc[i,'%OF_GROSS_EARNED_ABROAD'] = (df.loc[i, 'FOREIGN_GROSS'] / df.loc[i, 'WORLDWIDE_GROSS']) * 100

            for genre in movie.genre.split(','):
                genre = genre.strip()
                genre = genre.upper()
                df.loc[i, f"GENRE_{genre.replace(' ','_')}"] = True

            for col in df.columns:
                if col.startswith('GENRE_'):
                    df[col].fillna(False, inplace=True)

            for script_type in movie.script_type.split(','):
                script_type = script_type.strip()
                script_type = script_type.upper()
                df.loc[i, f"SCRIPT_{script_type.replace(' ','_')}"] = True

            for col in df.columns:
                if col.startswith('SCRIPT_'):
                    df[col].fillna(False, inplace=True)

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
