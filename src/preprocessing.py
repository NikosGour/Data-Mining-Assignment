import numpy as np
import pandas as pd


df = pd.read_excel('../data/movies.xlsx')
oscars = pd.read_csv('./../data/oscars.csv')

q = 0
for i, film in enumerate(df['Film']):
    if film in oscars['winners'].values:
        df.loc[i,'Oscar Winners'] = 'Oscar Winner'
        #BUG BELLOW RUN DEBUGGER
        df.loc[i,'Oscar Detail'] = oscars[oscars['winners'] == film]['category']
        q += 1
print(df['Oscar Winners'])
