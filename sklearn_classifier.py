import csv
import os, sys
from utils import *
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from termcolor import cprint
from collections import Counter
import matplotlib.pyplot as plt
from pprint import pprint, pformat
from us_state_abbrev import us_state_abbrev

from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from colnames import *
from load_dataset import load_and_preprocess, extract_features, split_categorical_and_interval

# from load_dataset import load_dataset

# X_train, X_test, y_train, y_test = load_dataset()

cprint('STARTING', 'white', 'on_green')

# nrows = 500
# nrows = 20000
nrows = None

N_JOBS = 2
N_ITER = 32

df_train = load_and_preprocess('TrainingSet(3).csv', nrows=nrows)
df_inference = load_and_preprocess('TestingSet(2).csv', nrows=nrows)

# Inference is the final prediction (without ground-truth)
X, y, inference_row_ids, inference_X, _, _, _ = extract_features(df_train, df_inference, colnames_categ, colnames_interval, fuzzy_matching=True, use_sentence_vec=True)

# Test is ONLY for evaluation metrics (contains ground-truth)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Create the SVC model object
print('Fitting the model...')
print("Tuning hyper-parameters")

# SVM
parameters_svc = {'kernel': ['rbf'], 'gamma': [1e-3, 0.01, 0.1, 0.2], 'C': [1, 10, 100, 1000]}
# parameters = {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

# Decision Tree
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 2000, stop = 8000, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(50, 110, num = 4)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
parameters_rf = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# clf1 = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters_svc, cv=3, n_jobs=16, verbose=5)
clf1 = RandomizedSearchCV(svm.SVC(decision_function_shape='ovr'), parameters_svc, n_iter=N_ITER, cv=3, n_jobs=N_JOBS, verbose=5)
clf2 = RandomizedSearchCV(RandomForestClassifier(), parameters_rf, n_iter=N_ITER, cv=3, n_jobs=N_JOBS, verbose=5)

for clf in [clf2]:
    clf.fit(X_train, y_train)
    # svc = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    print(means)
    # print(stds)

    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("Accuracy: {:.2%} ({:.2f}s) for {}".format(mean, std * 2, params))

    print('Predicting...')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
