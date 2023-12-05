import pandas as pd
import os

import Constants

oscars = pd.read_csv(os.path.join(Constants.ROOT_PROJECT_DIR, 'data/oscars.csv'))


def fill_oscars(df):
    for i, film in enumerate(df['TITLE']):
        if not df.loc[i, 'WON_OSCAR']:
            if film in oscars['winners'].values:
                df.loc[i, 'WON_OSCAR'] = True
                df.loc[i, 'OSCAR_DETAILS'] = oscars[oscars['winners'] == film]['category'].iloc[0]
    return df

def calculate_percentage_of_gross_earned_abroad(df):
    df['%OF_GROSS_EARNED_ABROAD'] = (df['FOREIGN_GROSS']/df['WORLDWIDE_GROSS']) * 100
    return df

def calculate_budget(df: pd.DataFrame):
    # The movie 'Life as We Know It' had a budget of `52.274.000.000$` according to the dataset.
    # This is obviously wrong, so we will fill the data with the correct budget of `38.000.000`
    # (https://www.google.com/search?q=Life+as+We+Know+It+budget&oq=life&gs_lcrp=EgZjaHJvbWUqCAgAEEUYJxg7MggIABBFGCcYOzIHCAEQLhiABDIHCAIQLhiABDIHCAMQLhiABDIHCAQQLhiABDIGCAUQRRg9MgYIBhBFGD0yBggHEEUYPdIBCDM2MzhqMGo5qAIAsAIA&sourceid=chrome&ie=UTF-8)
    df.loc[df.index[df['TITLE']=='Life as We Know It'][0],'BUDGET'] = 38
    df['BUDGET'] = df['BUDGET'].map(lambda x: x * 1_000_000)
    return df

def calculate_budget_recovered(df: pd.DataFrame):
    df['%BUDGET_RECOVERED'] = (df['WORLDWIDE_GROSS'] / df['BUDGET']) * 100
    return df

def calculate_budget_recovered_opening_weekend(df: pd.DataFrame):
    df['%BUDGET_RECOVERED_OPENING_WEEKEND'] = (df['OPENING_WEEKEND'] / df['BUDGET']) * 100
    return df

def calculate_average_critics(df: pd.DataFrame):
    df['AVERAGE_CRITICS'] = (df['RT_CRITICS'] + df['MC_CRITICS']) / 2
    return df

def calculate_average_audience(df: pd.DataFrame):
    df['AVERAGE_AUDIENCE'] = (df['RT_AUDIENCE'] + df['MC_AUDIENCE']) / 2
    return df

def calculate_rt_mc_difference(df: pd.DataFrame):
    df['RT_MC_AUDIENCE_DIFFERENCE'] = df['RT_AUDIENCE'] - df['MC_AUDIENCE']
    return df

def calculate_critics_audience_difference(df: pd.DataFrame):
    df['CRITICS_AUDIENCE_DIFFERENCE'] = df['AVERAGE_CRITICS'] - df['AVERAGE_AUDIENCE']
    return df
def fill_main(df):
    df = fill_oscars(df)
    df = calculate_percentage_of_gross_earned_abroad(df)
    df = calculate_budget(df)
    df = calculate_budget_recovered(df)
    df = calculate_budget_recovered_opening_weekend(df)
    df = calculate_average_critics(df)
    df = calculate_average_audience(df)
    df = calculate_rt_mc_difference(df)
    df = calculate_critics_audience_difference(df)
    return df