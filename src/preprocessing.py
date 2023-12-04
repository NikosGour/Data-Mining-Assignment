import numpy as np
import pandas as pd
import oscar_data_preprocessing.make_every_json_to_csv as utils
import Constants
df = pd.read_excel('../data/movies.xlsx')
oscars = pd.read_csv('../data/oscars.csv')

q = 0
for i, film in enumerate(df['Film']):
    if pd.isna(df.loc[i,'Oscar Winners']):
        if film in oscars['winners'].values:
            df.loc[i,'Oscar Winners'] = 'Oscar Winner'
            df.loc[i,'Oscar Detail'] = oscars[oscars['winners'] == film]['category'].iloc[0]
            q += 1


# utils.imdb_json_to_csv()
# utils.get_imdb_rating(df)
utils.quick_fix(df)