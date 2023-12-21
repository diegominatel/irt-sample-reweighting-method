# -*- coding: utf-8 -*-

''' General packages '''
import math
import numpy as np
import pandas as pd

''' Load classification algorithms form sklearn '''
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

''' class Mirt from mirt.py '''
from mirt import Mirt

def is_correct(true, predict):
    ''' Check correct and incorrect predictions
    Args:
        true: (list) true labels
        predict: (list) predicted target values
    Returns: (list) indicates correct and incorrect predictions
    '''
    if true == predict:
        return 1
    else:
        return 0     
    
is_correct = np.vectorize(is_correct)  


class Preprocessing_IRT():
    ''' Estimates the parameters of the items so that there is no DIF. Thus, using the abilities as a weight in the training of         classifiers
    ''' 
    def __init__(self):
        self.weights = None
        
        
    def execute(self, X, y):
        ''' Training the set of classifiers '''
        knn1 = KNeighborsClassifier(n_neighbors = 1, n_jobs=-1).fit(X, y)
        knn3 = KNeighborsClassifier(n_neighbors = 3, n_jobs=-1).fit(X, y)
        knn5 = KNeighborsClassifier(n_neighbors = 5, n_jobs=-1).fit(X, y)
        knn7 = KNeighborsClassifier(n_neighbors = 7, n_jobs=-1).fit(X, y)
        ''' Item modeling '''
        pred_knn3 = is_correct(y, knn3.predict(X))
        pred_knn1 = is_correct(y, knn1.predict(X))
        pred_knn5 = is_correct(y, knn5.predict(X))
        pred_knn7 = is_correct(y, knn7.predict(X))
        item_modeling = pd.DataFrame(np.transpose([pred_knn1, pred_knn3, pred_knn5, pred_knn7]))
        ''' Insert the groups in the item modeling '''
        group = list(X.index)
        ''' Insert the condition that all classifier have one incorrect and correct answer in both groups 
            (necessary to always run) '''
        group.append('Privileged')
        group.append('Privileged')
        group.append('Unprivileged')
        group.append('Unprivileged')
        item_modeling.loc[item_modeling.shape[0]] = [0]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 1] = [1]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 2] = [0]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 3] = [1]*item_modeling.shape[1]
        ''' Run Mirt '''
        mirt_model = Mirt()
        mirt_model.calculate(item_modeling, group)
        weights = mirt_model.estimated_abilities
        ''' remove the weights of insertion of line 56-63 '''
        n = len(X)
        weights = weights.drop([n, n + 1, n + 2, n + 3], axis=0)
        ''' Rescale weights ''' 
        return (1/MinMaxScaler(feature_range=(1, 5)).fit_transform(weights))*5