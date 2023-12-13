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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import torch

import Constants
import os
from src.preprocessing.preprocessing import Preprocessing
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.predict_class import PredictionMovie
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

preprocessing = Preprocessing()
df = pd.read_excel('data/movies.xlsx')
df = preprocessing.fit(df)
# print(df['WON_OSCAR'])
min_max_scaler = StandardScaler()

df['IMDB_RATING'] = min_max_scaler.fit_transform(df[['IMDB_RATING']])
df['OPENING_WEEKEND'] = min_max_scaler.fit_transform(df[['OPENING_WEEKEND']])
df['BUDGET'] = min_max_scaler.fit_transform(df[['BUDGET']])
df['WORLDWIDE_GROSS'] = min_max_scaler.fit_transform(df[['WORLDWIDE_GROSS']])
df['FOREIGN_GROSS'] = min_max_scaler.fit_transform(df[['FOREIGN_GROSS']])
df['DOMESTIC_GROSS'] = min_max_scaler.fit_transform(df[['DOMESTIC_GROSS']])


df = df.drop(columns=['TITLE'])
X = df.drop(columns=['WON_OSCAR'])
y = df['WON_OSCAR']

def decisionTree():
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    scores =[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        clf.fit(X.iloc[train_index], y.iloc[train_index])
        print(classification_report(y.iloc[test_index], clf.predict(X.iloc[test_index])))
        scores.append(clf.score(X.iloc[test_index], y.iloc[test_index]))
        # fig = plt.figure(figsize=(25, 20))
        # _ = sklearn.tree.plot_tree(clf, feature_names=X.columns, class_names=['False', 'True'], filled=True)
        # plt.savefig(f'tree{i}.svg', format='svg', bbox_inches='tight')

    print(f"{Fore.GREEN}Decision Tree Classifier{Style.RESET_ALL}")
    print(f"Accuracy: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")
    print('-' * 100)

    movie = PredictionMovie("Hachi: A Dog's Tale", 2009, "Original Screenplay", 64, 54, 85, 63,"Drama, Family", 108_382, 0, 47_707_417, 47_707_417	, 16, 8.1, 6, 13)
    movie = preprocessing.transform([movie])
    movie = movie.drop(columns=['TITLE','WON_OSCAR'])
    print(clf.predict_proba(movie))
    print(clf.coef_,clf.intercept_)
decisionTree()
def randomForest():
    clf = RandomForestClassifier(n_estimators=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    scores =[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        clf.fit(X.iloc[train_index], y.iloc[train_index])
        print(classification_report(y.iloc[test_index], clf.predict(X.iloc[test_index])))
        scores.append(clf.score(X.iloc[test_index], y.iloc[test_index]))
        fig = plt.figure(figsize=(25, 20))
        _ = sklearn.tree.plot_tree(clf.estimators_[0], feature_names=X.columns, class_names=['False', 'True'], filled=True)
        plt.savefig(f'tree{i}.svg', format='svg', bbox_inches='tight')

    print(f"{Fore.GREEN}Random Forest Classifier{Style.RESET_ALL}")
    print(f"Accuracy: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")
    print('-' * 100)

#XALIA GIA TO PROBLIMA
def KNN():
    clf = KNeighborsClassifier(n_neighbors=2)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    scores =[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        clf.fit(X.iloc[train_index], y.iloc[train_index])
        print(classification_report(y.iloc[test_index], clf.predict(X.iloc[test_index])))
        scores.append(clf.score(X.iloc[test_index], y.iloc[test_index]))

    print(f"{Fore.GREEN}KNeighbors Classifier{Style.RESET_ALL}")
    print(f"Accuracy: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")
    print('-' * 100)

# rfecv = RFECV(estimator=clf, cv=cv, scoring='accuracy')
# rfecv.fit(X, y)
#
# print(f"{Fore.GREEN}Decision Tree Classifier{Style.RESET_ALL}")
# print(f"Optimal number of features : {rfecv.n_features_}")
# print('-' * 100)
# print(f"{Fore.GREEN}Feature Ranking: {rfecv.ranking_}{Style.RESET_ALL}")
# print('-' * 100)
# print(f"{Fore.GREEN}Feature Support: {rfecv.support_}{Style.RESET_ALL}")
# print('-' * 100)
# print(f"{Fore.GREEN}Feature Importances: {rfecv.estimator_.feature_importances_}{Style.RESET_ALL}")
# print('-' * 100)
# print(f"{Fore.GREEN}Feature Names: {X.columns}{Style.RESET_ALL}")
# print('-' * 100)
#
# print(X.columns[rfecv.support_])
# print('-' * 100)
# columns = df.columns
data = PredictionMovie("PUSS IN BOOTS: THE LAST WISH", 2022, "Original Screenplay", 95, 73, 94, 88
                       , "Drama, Animation, Action", 12_400_000, 185_500_000, 299_200_000, 484_700_000,
                       110, 6.6, 12,22)
# new_df = pd.DataFrame(data=[data], columns=columns)
data = [data,data,data,data,data,data,data,data,data,data,data,data]
new_df = preprocessing.transform(data)
new_df = new_df.drop(columns=['TITLE','WON_OSCAR'])


# x = rfecv.predict(new_df)

# n_scores = len(rfecv.cv_results_["mean_test_score"])

# matplotlib.use('TkAgg')
# plt.scatter(df['WON_OSCAR'],df['RT_CRITICS'])
# plt.xlabel('WON_OSCAR')
# plt.ylabel('WORLDWIDE_GROSS')
# plt.show()

# dtree = DecisionTreeClassifier()
# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(dtree, X, y, cv=cv)
# print(f"{Fore.GREEN}Decision Tree Classifier{Style.RESET_ALL}")
# print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
# print('-'*100)
# print(f"{Fore.GREEN}Scores: {scores}{Style.RESET_ALL}")
# print('-'*100)
# print(f"{Fore.GREEN}Mean: {scores.mean()}{Style.RESET_ALL}")
# print('-'*100)
# print(f"{Fore.GREEN}Std: {scores.std()}{Style.RESET_ALL}")
# print('-'*100)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41, stratify=df['WON_OSCAR'])


# rtree = DecisionTreeRegressor()
# rforest = RandomForestRegressor(n_estimators=10,max_depth=None,min_samples_split=2)
# gauss = GaussianNB()
# model = SVC()


#
# dtree.fit(X_train, y_train)
#
# rtree.fit(X_train, y_train)
# rforest.fit(X_train, y_train)
# gauss.fit(X_train, y_train)
# model.fit(X_train, y_train)


# y_pred = dtree.predict(X_test)
#
# df = pd.DataFrame(dtree.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance',ascending=False)
#
# print(f"{Fore.GREEN}Decision Tree Classifier{Style.RESET_ALL}")
# print(classification_report(y_test, y_pred))
# print('-'*100)


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

# plt.show()
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
