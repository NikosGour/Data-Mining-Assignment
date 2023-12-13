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
    # We make everything uppercase and strip the whitespace for consistency.
    df['SCRIPT_TYPE'] = df['SCRIPT_TYPE'].apply(lambda x: x.upper())
    df['SCRIPT_TYPE'] = df['SCRIPT_TYPE'].apply(lambda x: x.strip())

    # We replace the value `REMAKE / REBOOT` with `REMAKE` because it's the same category and I prefer the naming.
    df['SCRIPT_TYPE'] = df['SCRIPT_TYPE'].apply(lambda x: x.replace('REMAKE / REBOOT', 'REMAKE'))

    # For every script type we create a new column and set it to True if the movie has that script type.
    # example: if script type = "REMAKE, SEQUEL, ADAPTATION" then we create 3 new columns
    # `SCRIPT_REMAKE`, `SCRIPT_SEQUEL`, `SCRIPT_ADAPTATION` and set them to True.
    # We also strip the whitespace for consistency.
    # if the column already exists we just set it to True.
    for i in df.index:
        for script in df.loc[i, 'SCRIPT_TYPE'].split(','):
            script = script.strip()
            df.loc[i, f"SCRIPT_{script.replace(' ', '_')}"] = True

    # We also drop the `SCRIPT_TYPE` column because we don't need it anymore.
    df.drop(columns=['SCRIPT_TYPE'], inplace=True)

    # We fill the NaN values with False.
    for col in df.columns:
        if col.startswith('SCRIPT_'):
            df[col] = df[col].fillna(False)
    return df


def clean_genre(df: pd.DataFrame):
    # We drop this movie because it's the only one with a genre of 'nan',
    # if we had way more movies with a genre of 'nan' we wouldn't drop the rows.
    try:
        df.drop(index=930, inplace=True)
    except Exception as e:
        print(e)
    df.index = pd.RangeIndex(0, len(df.index))

    # We make everything uppercase and strip the whitespace for consistency.
    df['GENRE'] = df['GENRE'].apply(lambda x: x.upper())
    df['GENRE'] = df['GENRE'].apply(lambda x: x.strip())

    # There is some rows where there is a dot instead of a comma, so we replace them.
    df['GENRE'] = df['GENRE'].apply(lambda x: x.replace('.', ','))

    df['GENRE']  = df['GENRE'].apply(lambda x: x.rstrip(','))

    # There are some rows where there is a typo `FAMIILY` instead of `FAMILY`
    df['GENRE'] = df['GENRE'].apply(lambda x: x.replace('II', 'I'))

    # There are some rows where there is a typo `THRILER` instead of `THRILLER`
    df['GENRE'] = df['GENRE'].apply(lambda x: x.replace('THRILER', 'THRILLER'))

    # There are some rows where there is the value `SPORTS` and some with `SPORT`, we combine them into one category.
    df['GENRE'] = df['GENRE'].apply(lambda x: x.replace('SPORTS', 'SPORT'))

    # There are some rows where there is the value `MUSICAL` and some with `MUSIC`, we combine them into one category.
    df['GENRE'] = df['GENRE'].apply(lambda x: x.replace('MUSICAL', 'MUSIC'))
    # I prefer the name `MUSICAL` over `MUSIC`, so I replace it back. Now all the movies with genre `MUSIC` are `MUSICAL`
    # and all the movies with genre `MUSICAL` are still `MUSICAL`.
    df['GENRE'] = df['GENRE'].apply(lambda x: x.replace('MUSIC', 'MUSICAL'))

    # There are some rows where there is a typo `HORRO` instead of `HORROR`
    # replacing it `HORROR` with `HORRO` and then back to `HORROR` is a
    # quick fix to avoid more complicated code. The result is
    # all the movies that had either `HORRO` or `HORROR` are now `HORROR`.
    df['GENRE'] = df['GENRE'].apply(lambda x: x.replace('HORROR', 'HORRO'))
    df['GENRE'] = df['GENRE'].apply(lambda x: x.replace('HORRO', 'HORROR'))

    # For every script type we create a new column and set it to True if the movie has that script type.
    # example: if script type = "COMEDY, DRAMA, ROMANCE" then we create 3 new columns
    # `GENRE_COMEDY`, `GENRE_DRAMA`, `GENRE_ROMANCE` and set them to True.
    # We also strip the whitespace for consistency.
    # if the column already exists we just set it to True.
    for i in df.index:
        for genre in df.loc[i, 'GENRE'].split(','):
            genre = genre.strip()
            df.loc[i, f"GENRE_{genre.replace(' ', '_')}"] = True

    # We also drop the `GENRE` column because we don't need it anymore.
    df.drop(columns=['GENRE'], inplace=True)

    # We fill the NaN values with False.
    for col in df.columns:
        if col.startswith('GENRE'):
            df[col] = df[col].fillna(False)
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
