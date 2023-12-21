# -*- coding: utf-8 -*-

''' General packages '''
import pandas as pd
import numpy as np

''' Load fairness metrics '''
from classification_fairness_measures import (Measures, MeasuresFairness, get_group_fairness_measure_names,
                                              get_all_measure_names, get_individual_fairness_measure_names,
                                              get_performance_measure_names)

''' Load from sklearn '''
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

class Validation:
    def __init__(self, stratified=True, stratified_by='group_target', metric='fairness_performance', 
                 shuffle=False, random_state=None):
        self.stratified = stratified
        self.stratified_by = stratified_by
        self.metric = metric
        self.shuffle = shuffle
        self.random_state = random_state
        self.fairness = False
        self.measures = None
        self.scores = None
        self.groupby_scores = None
        self.checks()
    
    def checks(self):
        
        if not self.stratified_by in _StratifiedBy._getmethods():
            raise ValueError('Parameter \'stratified_by\' does not allow the following value: -- %s --.' % (self.stratified_by))
            
        if not self.metric in ['fairness_performance', 'only_performance']:
            raise ValueError('Input \'fairness_performance\' or \'only_performance\' in the metric parameter.''')

        if not self.stratified:
            self.stratified = None

        if not self.shuffle:
            self.random_state = None
            
        if self.metric == 'fairness_performance':
            self.measures = MeasuresFairness()
            self.fairness = True
        else:
            self.measures = Measures()
            self.fairness = False
    
    def _initialize_reports(self):
        self.scores = None
        self.groupby_scores = None
    
    def get_scores_columns(self):
        if self.fairness:
            return get_all_measure_names()
        else:
            return get_performance_measure_names()
    
    def get_groupby_scores_columns(self):
        return get_performance_measure_names()
    
    def get_groupfairness_scores(self):
        if self.fairness == True:
            return self.scores[get_group_fairness_measure_names()]
    
    def get_individualfairness_scores(self):
        if self.fairness == True:
            return self.scores[get_individual_fairness_measure_names()]
    
    def get_performance_scores(self):
        return self.scores[get_performance_measure_names()]
    
    def get_ratio_scores(self):
        if self.fairness == True:
            columns = list(map(lambda x: 'ratio_' + x, get_performance_measure_names()))
            return self.scores[columns]
    
    def get_difference_scores(self):
        if self.fairness == True:
            columns = list(map(lambda x: 'dif_' + x, get_performance_measure_names()))
            return self.scores[columns]      
    
class Holdout(Validation):
    
    def __init__(self, test_size=0.25, stratified=True, stratified_by='group_target', metric='fairness_performance', 
                 shuffle=False, random_state=None):
        
        super().__init__(stratified, stratified_by, metric, shuffle, random_state)
        self.test_size = test_size
    
    def get_validation_name(self):
        return 'holdout'
        
    def _update_reports(self):
        self.scores = self.measures.scores
        if self.fairness == True:
            self.groupby_scores = self.measures.groupby_scores
        
    def calculate(self, x, y, clf, priv_group=None):
        self._initialize_reports()
        by = getattr(_StratifiedBy, self.stratified_by)(x, y) if self.stratified else None
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=by, test_size=self.test_size,
                                                            random_state=self.random_state)
        
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        if self.fairness == True:
            self.measures.calculate(y_test, y_predict, x_test.index, priv_group)
        else:
            self.measures.calculate(y_test, y_predict)
        self._update_reports()
    

class NFold(Validation):
    
    def __init__(self, n=10, stratified=True, stratified_by='group_target', metric='fairness_performance', shuffle=False,
                 random_state=None):
        
        super().__init__(stratified, stratified_by, metric, shuffle, random_state)
        ''' Unique parameters NFold '''
        self.n = n
        self.folds_scores = None
        self.folds_groupby_scores = None
        self.scores_std = None
        self.groupby_scores_std = None
        Fold = StratifiedKFold if stratified else KFold
        self.kf = Fold(n_splits=n, shuffle=shuffle, random_state=random_state)
        
    def get_validation_name(self):
        return 'nfold'
        
    def _update_reports(self):
        self.folds_scores = pd.concat([self.folds_scores, self.measures.scores], ignore_index=True)
        if self.fairness == True:
            self.folds_groupby_scores = pd.concat([self.folds_groupby_scores, self.measures.groupby_scores], ignore_index=True)
            
    def _finish_reports(self):
        self.scores = self.folds_scores.mean().to_frame().T
        self.scores_std = self.folds_scores.std().to_frame().T
        if self.fairness == True:
            columns = get_performance_measure_names()
            self.folds_groupby_scores[columns] = self.folds_groupby_scores[columns].astype('float64')
            self.groupby_scores = self.folds_groupby_scores.groupby(by='group').mean()
            self.groupby_scores_std = self.folds_groupby_scores.groupby(by='group').std()  
        
    def calculate(self, x, y, clf, priv_group=None):
        self._initialize_reports()
        by = getattr(_StratifiedBy, self.stratified_by)(x, y) if self.stratified else None
        for i, (train_index, test_index) in enumerate(self.kf.split(x, by)):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = clf.set_params(**clf.get_params())
            clf.fit(x_train, y_train)
            y_predict = clf.predict(x_test)
            if self.fairness == True:
                self.measures.calculate(y_test, y_predict, x_test.index, priv_group)
            else:
                self.measures.calculate(y_test, y_predict)
            self._update_reports()  
        self._finish_reports()

class _StratifiedBy():
    @staticmethod
    def _getmethods():
        return [method for method in dir(_StratifiedBy) if method.startswith('_') is False]

    @staticmethod
    def _get_group(x):
        if isinstance(x.index, pd.MultiIndex):
            return LabelEncoder().fit_transform(list(map('/'.join, list(x.index))))
        else:
            return LabelEncoder().fit_transform(list(x.index))

    @staticmethod
    def _get_group_target(x, y):
        groups = _StratifiedBy._get_group(x)
        group_target = [str(group) + str(target) for group, target in zip(groups, y)]
        return LabelEncoder().fit_transform(group_target)
    
    @staticmethod
    def none(x, y):
        return None

    @staticmethod
    def target(x, y):
        return y

    @staticmethod
    def group(x, y):
        return _StratifiedBy._get_group(x)

    @staticmethod
    def group_target(x, y):
        return _StratifiedBy._get_group_target(x, y)