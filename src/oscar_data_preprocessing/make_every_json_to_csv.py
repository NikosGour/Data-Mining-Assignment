i = 0

import os
import json
import pandas as pd


def oscars_dataframe():

    # data taken from https://github.com/AminFadaee/Academy-Awards-Data
    data_folder = './winners_only'
    files = os.listdir(data_folder)
    files.sort()
    df = pd.DataFrame(columns=['year', 'winners', 'category'])
    for file_name in files:
        path = os.path.join(data_folder, file_name)
        file_name_without_extension = file_name.split('.')[0]
        with open(path) as f:
            data = json.load(f)
            for obj in data:
                if len(obj['winners']) == 2:
                    df.loc[len(df)] = [file_name_without_extension, obj['winners'][0], obj['category']]
                    df.loc[len(df)] = [file_name_without_extension, obj['winners'][1], obj['category']]
                else:
                    df.loc[len(df)] = [file_name_without_extension, obj['winners'][0], obj['category']]


    df.to_csv('../../data/oscars.csv', index=False)
oscars_dataframe()
