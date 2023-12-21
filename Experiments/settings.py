import math
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def set_configs(n_columns):
    
    ada = {
        'ADA' : [XGBClassifier,
                 {'n_estimators' : list(range(100, 475, 25)),
                  'n_jobs' : [-1], 'random_state' : [42]}]
    }
    
    decision_tree = {
        'DT' : [DecisionTreeClassifier,
                {'criterion' : ['gini'],
                 'min_samples_leaf' : [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                 'min_samples_split' : [5],
                 'random_state' : [42]}]
    }
    
    random_forest = {
        'RF' : [RandomForestClassifier,
                {'n_estimators' : list(range(100, 475, 25)),
                 'min_samples_split' : [math.floor(abs(math.sqrt(n_columns - 1)))], 
                 'n_jobs' : [-1], 'random_state' : [42]}]
    }
    
    svm = {
        'SVM' : [SVC,
                 {'kernel' : ['rbf'], 'C' : [1], 'gamma' : list(np.arange(0.0025, 1.075, 0.075)), 
                  'random_state' : [42]}]
    }
    
    xgb = {
        'XGB' : [XGBClassifier,
                 {'n_estimators' : list(range(100, 475, 25)),
                  'random_state' : [42]}]
    }
    
    
    all_configs = {
        'ada'   : ada,
        'dt'    : decision_tree,
        'rf'    : random_forest,
        'svm'   : svm,
        'xgb'   : xgb
    }
    
    return all_configs