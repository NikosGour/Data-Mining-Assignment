import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.tree
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import Constants
import os
import src.preprocessing.preprocessing as preprocessing
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('data/movies.xlsx')
df = preprocessing.setup(df)
# print(df['WON_OSCAR'])

min_max_scaler = MinMaxScaler()

# df['IMDB_RATING'] = min_max_scaler.fit_transform(df[['IMDB_RATING']])
# df['OPENING_WEEKEND'] = min_max_scaler.fit_transform(df[['OPENING_WEEKEND']])
# df['BUDGET'] = min_max_scaler.fit_transform(df[['BUDGET']])
# df['WORLDWIDE_GROSS'] = min_max_scaler.fit_transform(df[['WORLDWIDE_GROSS']])
# df['FOREIGN_GROSS'] = min_max_scaler.fit_transform(df[['FOREIGN_GROSS']])
# df['DOMESTIC_GROSS'] = min_max_scaler.fit_transform(df[['DOMESTIC_GROSS']])




non_oscar = df[df['WON_OSCAR'] == False].sample(n=91,axis=0)
oscar = df[df['WON_OSCAR'] == True]

df = pd.concat([non_oscar,oscar])
df = df.drop(columns=['TITLE'])
X = df.drop(columns=['WON_OSCAR'])
y = df['WON_OSCAR']


# matplotlib.use('TkAgg')
# plt.scatter(df['WON_OSCAR'],df['RT_CRITICS'])
# plt.xlabel('WON_OSCAR')
# plt.ylabel('WORLDWIDE_GROSS')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41, stratify=df['WON_OSCAR'])


dtree = DecisionTreeClassifier()
# rtree = DecisionTreeRegressor()
# rforest = RandomForestRegressor(n_estimators=10,max_depth=None,min_samples_split=2)
# gauss = GaussianNB()
# model = SVC()



dtree.fit(X_train, y_train)
# rtree.fit(X_train, y_train)
# rforest.fit(X_train, y_train)
# gauss.fit(X_train, y_train)
# model.fit(X_train, y_train)


y_pred = dtree.predict(X_test)

df = pd.DataFrame(dtree.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance',ascending=False)

print(f"{Fore.GREEN}Decision Tree Classifier{Style.RESET_ALL}")
print(classification_report(y_test, y_pred))
print('-'*100)


# y_pred = rtree.predict(X_test)
# print(f"{Fore.GREEN}Decision Tree Regressor{Style.RESET_ALL}")
# print(classification_report(y_test, y_pred))
# print('-'*100)

# y_pred = rforest.predict(X_test)
# print(f"{Fore.GREEN}Random Forest Regressor{Style.RESET_ALL}")
# print(classification_report(y_test, y_pred))
# print('-'*100)
#
# y_pred = gauss.predict(X_test)
# print(f"{Fore.GREEN}Gaussian Naive Bayes{Style.RESET_ALL}")
# print(classification_report(y_test, y_pred))
# print('-'*100)
# #
# # y_pred = model.predict(X_test)
# # print(f"{Fore.GREEN}Support Vector Machine{Style.RESET_ALL}")
# # print(classification_report(y_test, y_pred))
# # print('-'*100)
#
fig = plt.figure(figsize=(25, 20))
_ = sklearn.tree.plot_tree(dtree, feature_names=X.columns, class_names=['False', 'True'], filled=True)
plt.savefig('tree.svg', format='svg', bbox_inches='tight')
# # plt.show()
#

# movie_name = 'WALL-E'
# url = f"https://www.omdbapi.com/?t={movie_name}&apikey=10461461"
#
#
# response = requests.get(url)
#
# print(response.json()['imdbRating'])

# df1 = pd.read_csv('data/imdb_ratings.csv')
# df2 = pd.read_csv('data/not_found.csv')
#
# print(len(df1)+len(df2) == len(df))

