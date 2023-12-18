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

matplotlib.use('TkAgg')
warnings.simplefilter(action='ignore', category=FutureWarning)

preprocessing = Preprocessing()
df = pd.read_excel('data/movies.xlsx')
df = preprocessing.fit(df)
# print(df['WON_OSCAR'])


df = df.drop(columns=['TITLE'])
X = df.drop(columns=['WON_OSCAR'])
y = df['WON_OSCAR']



    # movie = PredictionMovie("Hachi: A Dog's Tale", 2009, "Original Screenplay", 64, 54, 85, 63, "Drama, Family",
    #                         108_382, 0, 47_707_417, 47_707_417, 16, 8.1, 6, 13)
    # movie = preprocessing.transform([movie])
    # movie = movie.drop(columns=['TITLE', 'WON_OSCAR'])
    # print(clf.predict_proba(movie))


# decisionTree()

#
# movie = PredictionMovie("Hachi: A Dog's Tale", 2009, "Original Screenplay", 64, 54, 85, 63, "Drama, Family",
#                         108_382, 0, 47_707_417, 47_707_417, 16, 8.1, 6, 13)
# movie = preprocessing.transform([movie])
# movie = movie.drop(columns=['TITLE', 'WON_OSCAR'])
# print(clf.predict_proba(movie))
#


from sklearn.metrics import roc_curve , precision_recall_curve

# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=df['WON_OSCAR'])
# false_class_weight = 0.999
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=1000)
knn = KNeighborsClassifier(n_neighbors=3)
lr = LogisticRegression()
gnb = GaussianNB()
svc = SVC(probability=True)

cv = StratifiedKFold(n_splits=5, shuffle=True)
for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    train_X = X.iloc[train_index]
    train_y = y.iloc[train_index]
    test_X = X.iloc[test_index]
    test_y = y.iloc[test_index]

    dt.fit(train_X, train_y)
    rf.fit(train_X, train_y)
    knn.fit(train_X, train_y)
    lr.fit(train_X, train_y)
    gnb.fit(train_X, train_y)
    svc.fit(train_X, train_y)

    r_probs = [0 for _ in range(len(test_y))]
    dt_probs = dt.predict_proba(test_X)
    rf_probs = rf.predict_proba(test_X)
    knn_probs = knn.predict_proba(test_X)
    lr_probs = lr.predict_proba(test_X)
    gnb_probs = gnb.predict_proba(test_X)
    svc_probs = svc.predict_proba(test_X)

    dt_probs = dt_probs[:, 1]
    rf_probs = rf_probs[:, 1]
    knn_probs = knn_probs[:, 1]
    lr_probs = lr_probs[:, 1]
    gnb_probs = gnb_probs[:, 1]
    svc_probs = svc_probs[:, 1]
#
# r_auc = roc_auc_score(test_y, r_probs)
# dt_auc = roc_auc_score(test_y, dt_probs)
# rf_auc = roc_auc_score(test_y, rf_probs)
# knn_auc = roc_auc_score(test_y, knn_probs)
# lr_auc = roc_auc_score(test_y, lr_probs)
# gnb_auc = roc_auc_score(test_y, gnb_probs)
# svc_auc = roc_auc_score(test_y, svc_probs)
#
# print(f"Random (chance) Prediction: AUROC = {r_auc:.3f}")
# print(f"Decision Tree: AUROC = {dt_auc:.3f}")
# print(f"Random Forest: AUROC = {rf_auc:.3f}")
# print(f"K Nearest Neighbors: AUROC = {knn_auc:.3f}")
# print(f"Logistic Regression: AUROC = {lr_auc:.3f}")
# print(f"Gaussian Naive Bayes: AUROC = {gnb_auc:.3f}")
# print(f"Support Vector Classifier: AUROC = {svc_auc:.3f}")


    r_precision, r_recall, _ = precision_recall_curve(test_y, r_probs)
    dt_precision, dt_recall, _ = precision_recall_curve(test_y, dt_probs)
    rf_precision, rf_recall, _ = precision_recall_curve(test_y, rf_probs)
    knn_precision, knn_recall, _ = precision_recall_curve(test_y, knn_probs)
    lr_precision, lr_recall, _ = precision_recall_curve(test_y, lr_probs)
    gnb_precision, gnb_recall, _ = precision_recall_curve(test_y, gnb_probs)
    svc_precision, svc_recall, _ = precision_recall_curve(test_y, svc_probs)

    plt.figure(figsize=(10, 10))
    plt.plot(r_recall, r_precision, linestyle='--', label='Random prediction' )
    plt.plot(dt_recall, dt_precision, marker='.', label='Decision Tree')
    plt.plot(rf_recall, rf_precision, marker='.', label='Random Forest' )
    plt.plot(knn_recall, knn_precision, marker='.', label='KNN')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic Regression')
    plt.plot(gnb_recall, gnb_precision, marker='.', label='Gaussian Naive Bayes')
    plt.plot(svc_recall, svc_precision, marker='.', label='Support Vector Classifier')

    plt.title('Recall Precision Plot')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.savefig(f'pr{i}.svg', format='svg', bbox_inches='tight')

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
                       110, 6.6, 12, 22)
# new_df = pd.DataFrame(data=[data], columns=columns)
data = [data, data, data, data, data, data, data, data, data, data, data, data]
new_df = preprocessing.transform(data)
new_df = new_df.drop(columns=['TITLE', 'WON_OSCAR'])

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
