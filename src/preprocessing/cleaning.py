import pandas as pd

def clean_worldwide_gross(df):
    df['WORLDWIDE_GROSS'] = df['WORLDWIDE_GROSS'].apply(
        lambda x: float(x) if isinstance(x, int) or isinstance(x, float) else x.replace(',', ''))
    df = df.astype({'WORLDWIDE_GROSS': float})
    return df


def clean_foreign_gross(df):
    df['FOREIGN_GROSS'] = df['FOREIGN_GROSS'].apply(
        lambda x: float(x) if isinstance(x, int) or isinstance(x, float) else x.replace(',', ''))
    df = df.astype({'FOREIGN_GROSS': float})
    return df

def clean_oscar_winners_filter(x):
    if not pd.isna(x):
        if x == 'Oscar winner' or x == 'Oscar Winner':
            return True
    return False

def clean_oscar_winners(df):
    df['WON_OSCAR'] = df['WON_OSCAR'].map(lambda x: clean_oscar_winners_filter(x))
    return df

def clean_general_float(x):
    if isinstance(x, str):
        if x == '-':
            return pd.NA

        x = x.replace(',', '')
    return float(x)

def clean_budget(df):
    df['BUDGET'] = df['BUDGET'].apply(lambda x: clean_general_float(x))
    df['BUDGET'] = df['BUDGET'].fillna(df['BUDGET'].mean())
    df = df.astype({'BUDGET': float})
    return df

def clean_opening_weekend(df):
    df['OPENING_WEEKEND'] = df['OPENING_WEEKEND'].apply(lambda x: clean_general_float(x))
    return df

def clean_rt_critics(df):
    df['RT_CRITICS'] = df['RT_CRITICS'].apply(lambda x: clean_general_float(x))
    df['RT_CRITICS'] = df['RT_CRITICS'].fillna(0)
    df['RT_CRITICS'] = df['RT_CRITICS'].apply(lambda x: int(x))
    return df

def clean_mc_critics(df: pd.DataFrame):
    df['MC_CRITICS'] = df['MC_CRITICS'].apply(lambda x: clean_general_float(x))
    df['MC_CRITICS'] = df['MC_CRITICS'].fillna(0)
    df['MC_CRITICS'] = df['MC_CRITICS'].apply(lambda x: int(x))
    return df

def clean_rt_audience(df: pd.DataFrame):
    df['RT_AUDIENCE'] = df['RT_AUDIENCE'].apply(lambda x: clean_general_float(x))
    df['RT_AUDIENCE'] = df['RT_AUDIENCE'].fillna(0)
    df['RT_AUDIENCE'] = df['RT_AUDIENCE'].apply(lambda x: int(x))
    return df

def clean_mc_audience(df: pd.DataFrame):
    df['MC_AUDIENCE'] = df['MC_AUDIENCE'].apply(lambda x: clean_general_float(x))
    df['MC_AUDIENCE'] = df['MC_AUDIENCE'].fillna(0)
    df['MC_AUDIENCE'] = df['MC_AUDIENCE'].apply(lambda x: int(x))
    return df
def clean_main(df):
    df = clean_foreign_gross(df)
    df = clean_worldwide_gross(df)
    df = clean_oscar_winners(df)
    df = clean_budget(df)
    df = clean_opening_weekend(df)
    df = clean_rt_critics(df)
    df = clean_mc_critics(df)
    df = clean_rt_audience(df)
    df = clean_mc_audience(df)
    return df
