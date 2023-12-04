# TODO: NOT DONE AT ALL , WHOLE FILE NEEDS LOOKING INTO
import numpy as np
import pandas as pd
import oscar_data_preprocessing.create_additional_data as utils
import Constants
import os
df = pd.read_excel('../data/movies.xlsx')
oscars = pd.read_csv('../data/oscars.csv')

q = 0
for i, film in enumerate(df['Film']):
    if pd.isna(df.loc[i,'Oscar Winners']):
        if film in oscars['winners'].values:
            df.loc[i,'Oscar Winners'] = 'Oscar Winner'
            df.loc[i,'Oscar Detail'] = oscars[oscars['winners'] == film]['category'].iloc[0]
            q += 1


utils.create_enhanced_movie_data_csv(df)
