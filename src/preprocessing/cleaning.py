import pandas as pd


def clean_general_float(x):
    if isinstance(x, str):
        if x == '-':
            return pd.NA

        x = x.replace(',', '')
    return float(x)


def clean_worldwide_gross(df):
    df['WORLDWIDE_GROSS'] = df['WORLDWIDE_GROSS'].apply(lambda x: clean_general_float(x))
    df = df.astype({'WORLDWIDE_GROSS': float})
    return df


def clean_foreign_gross(df):
    df['FOREIGN_GROSS'] = df['FOREIGN_GROSS'].apply(lambda x: clean_general_float(x))
    df = df.astype({'FOREIGN_GROSS': float})
    return df

def clean_domestic_gross(df):
    df['DOMESTIC_GROSS'] = df['DOMESTIC_GROSS'].apply(lambda x: clean_general_float(x))
    df = df.astype({'DOMESTIC_GROSS': float})
    return df
def clean_oscar_winners_filter(x):
    if not pd.isna(x):
        if x == 'Oscar winner' or x == 'Oscar Winner':
            return True
    return False


def clean_oscar_winners(df):
    df['WON_OSCAR'] = df['WON_OSCAR'].map(lambda x: clean_oscar_winners_filter(x))
    return df


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
    df['RT_CRITICS'] = df['RT_CRITICS'].fillna(df['RT_CRITICS'].mean())
    df['RT_CRITICS'] = df['RT_CRITICS'].apply(lambda x: int(x))
    return df


def clean_mc_critics(df: pd.DataFrame):
    df['MC_CRITICS'] = df['MC_CRITICS'].apply(lambda x: clean_general_float(x))
    df['MC_CRITICS'] = df['MC_CRITICS'].fillna(df['MC_CRITICS'].mean())
    df['MC_CRITICS'] = df['MC_CRITICS'].apply(lambda x: int(x))
    return df


def clean_rt_audience(df: pd.DataFrame):
    df['RT_AUDIENCE'] = df['RT_AUDIENCE'].apply(lambda x: clean_general_float(x))
    df['RT_AUDIENCE'] = df['RT_AUDIENCE'].fillna(df['RT_AUDIENCE'].mean())
    df['RT_AUDIENCE'] = df['RT_AUDIENCE'].apply(lambda x: int(x))
    return df


def clean_mc_audience(df: pd.DataFrame):
    df['MC_AUDIENCE'] = df['MC_AUDIENCE'].apply(lambda x: clean_general_float(x))
    df['MC_AUDIENCE'] = df['MC_AUDIENCE'].fillna(df['MC_AUDIENCE'].mean())
    df['MC_AUDIENCE'] = df['MC_AUDIENCE'].apply(lambda x: int(x))
    return df


def clean_script_type(df: pd.DataFrame):
    # Some values are the same , but in different order, so I replace them with the other one.
    df.loc[df['SCRIPT_TYPE'] == 'adaptation, sequel', 'SCRIPT_TYPE'] = 'sequel, adaptation'
    df.loc[df['SCRIPT_TYPE'] == 'remake / reboot, adaptation', 'SCRIPT_TYPE'] = 'adaptation, remake / reboot'
    df.loc[df['SCRIPT_TYPE'] == 'remake', 'SCRIPT_TYPE'] = 'remake / reboot'
    df.loc[df['SCRIPT_TYPE'] == 'based on a true story, adaptation', 'SCRIPT_TYPE'] \
        = 'adaptation, based on a true story'

    df['SCRIPT_TYPE'] = df['SCRIPT_TYPE'].apply(lambda x: str(x).strip())
    return df


def clean_genre(df: pd.DataFrame):
    # We drop this movie because it's the only one with a genre of 'nan',
    # if we had way more movies with a genre of 'nan' we wouldn't drop the rows.
    df.drop(index=930, inplace=True)
    df.index = pd.RangeIndex(0, len(df.index))
    df['GENRE'] = df['GENRE'].apply(lambda x: str(x).strip())
    return df


def clean_column_values(df):
    df = clean_foreign_gross(df)
    df = clean_worldwide_gross(df)
    df = clean_domestic_gross(df)
    df = clean_oscar_winners(df)
    df = clean_budget(df)
    df = clean_opening_weekend(df)
    df = clean_rt_critics(df)
    df = clean_mc_critics(df)
    df = clean_rt_audience(df)
    df = clean_mc_audience(df)
    df = clean_script_type(df)
    df = clean_genre(df)
    return df
