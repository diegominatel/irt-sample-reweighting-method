# -*- coding: utf-8 -*-

''' Remember: 
    selection_rate (difference)                == statistical_parity_difference
    selection_rate (ratio)                     == disparate_impact_ratio
    recall (tpr) (difference)                  == equal_oportunity_difference
    (fpr (difference) + recall (difference))/2 == average_odds_difference
    
    ratio = unprivileged/privileged
    difference = unprivileged - privileged
'''

''' General packages '''
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

''' Measures assistant '''
from aif360.sklearn.metrics import difference, ratio
''' Group fairness measures '''
from aif360.sklearn.metrics import statistical_parity_difference, disparate_impact_ratio, average_odds_difference, equal_opportunity_difference, between_group_generalized_entropy_error
''' Individual fairness measures '''
from aif360.sklearn.metrics import theil_index, coefficient_of_variation, generalized_entropy_error

class _ConfusionMatrixMeasures:
    """
    Class that contains all performance measures for binary classification.
    """
    @staticmethod
    def _getmethods():
        """
        Returns: all performance measure names
        """
        return [method for method in dir(_ConfusionMatrixMeasures) if method.startswith('_') is False]
    
    @staticmethod
    def accuracy(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return (tp+tn)/(tn+tp+fn+fp)
    
    @staticmethod
    def tpr(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return tp/(tp+fn)
    
    @staticmethod
    def tnr(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return tn/(tn+fp)
    
    @staticmethod
    def ppv(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return tp/(tp+fp)
    
    @staticmethod
    def npv(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return tn/(tn+fn)
    
    @staticmethod
    def fnr(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return fn/(fn+tp)
    
    @staticmethod
    def fpr(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return fp/(fp+tn)
    
    @staticmethod
    def fdr(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return fp/(fp+tp)
    
    @staticmethod
    def for_score(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return fn/(fn+tn)
    
    @staticmethod
    def balanced_accuracy(y_true, y_predict):
        return (_ConfusionMatrixMeasures.tpr(y_true, y_predict) + _ConfusionMatrixMeasures.tnr(y_true, y_predict))/2
    
    @staticmethod
    def f1(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return (2*tp)/((2*tp)+fp+fn)
    
    @staticmethod
    def f1_macro(y_true, y_predict):
        return f1_score(y_true, y_predict, average='macro')
    
    @staticmethod
    def f1_micro(y_true, y_predict):
        return f1_score(y_true, y_predict, average='micro')

    @staticmethod
    def selection_rate(y_true, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        return (tp+fp)/(tn+fp+fn+tp)
    
_performance_measures = {
    'accuracy' : _ConfusionMatrixMeasures.accuracy,
    'balanced_accuracy' : _ConfusionMatrixMeasures.balanced_accuracy,
    'f1' : _ConfusionMatrixMeasures.f1,
    'f1_macro' : _ConfusionMatrixMeasures.f1_macro,
    'f1_micro' : _ConfusionMatrixMeasures.f1_micro,
    'selection_rate' : _ConfusionMatrixMeasures.selection_rate,
    'recall' : _ConfusionMatrixMeasures.tpr, # recall is the same as tpr
    'precision' : _ConfusionMatrixMeasures.ppv, # precision is the same as ppv
    'fpr' : _ConfusionMatrixMeasures.fpr,
    'fnr' : _ConfusionMatrixMeasures.fnr,
    'npv' : _ConfusionMatrixMeasures.npv,
    'tnr' : _ConfusionMatrixMeasures.tnr,
    'fdr' : _ConfusionMatrixMeasures.fdr,
    'for' : _ConfusionMatrixMeasures.for_score
}

_group_fairness_measures = {
    'statistical_parity_difference': statistical_parity_difference,
    'disparate_impact_ratio' : disparate_impact_ratio,
    'equal_oportunity_difference' : equal_opportunity_difference, 
    'average_odds_difference' : average_odds_difference,
    'group_theil_index' : between_group_generalized_entropy_error
}

_individual_fairness_measures = {
    'theil_index' : theil_index, 
    'coefficient_variation' : coefficient_of_variation
}

_assist_measures = {
    'dif' : difference,
    'ratio' : ratio
}

def get_performance_measure_names():
    return list(_performance_measures.keys())

def get_group_fairness_measure_names():
    return list(_group_fairness_measures.keys())

def get_individual_fairness_measure_names():
    return list(_individual_fairness_measures.keys())

def get_ratio_measure_names():
    return list(map(lambda x: 'ratio_' + x, get_performance_measure_names()))
    
def get_difference_measure_names():
    return list(map(lambda x: 'dif_' + x, get_performance_measure_names()))

def get_assist_measures():
    return list(_assist_measures.keys())

def get_all_measure_names():
    names = get_performance_measure_names() + get_group_fairness_measure_names() + get_individual_fairness_measure_names()
    for prefix in get_assist_measures():
        ''' assist measures only apply to performance measures '''
        names = names + list(map(lambda x: prefix + '_' + x, get_performance_measure_names()))
    return names

def convert_index(l, privileged_group):
    ''' Function that converts the index value to numeric, necessary to use AIF360
    '''
    if l == privileged_group:
        return 1
    else:
        return 0
    
def array_to_entropy(y_true, y_predict):
    return y_predict - y_true + 1

convert_index = np.vectorize(convert_index)
array_to_entropy = np.vectorize(array_to_entropy)

class Measures:
    """ Calculates all performance measures
    """
    def __init__(self):
        self.scores = None
        
    def _initialize_reports(self):
        self.scores = pd.DataFrame(columns = get_performance_measure_names())
        
    def _scores(self, y_true, y_predict):
        for name, method in _performance_measures.items():
            self.scores.loc['score', name] = method(y_true, y_predict)     
    
    def calculate(self, y_true, y_predict):
        self._initialize_reports()
        self._scores(y_true, y_predict)

class MeasuresFairness(Measures):
    """ Calculates all performance and fairness measures (privileged and unprivileged group)
    """
    def __init__(self):        
        self.scores = None
        self.privileged_group = None
        self.groupby_scores = None
        
    def _initialize_reports(self):
        self.scores = pd.DataFrame(columns = get_all_measure_names())
        self.groupby_scores = pd.DataFrame(columns = ['group'] + get_performance_measure_names())
        
    def _update_scores(self, y_true, y_predict):
        b = array_to_entropy(y_true['class'], y_predict) # this is to calculate theil index
        ''' calculate performance measures '''
        for name, method in _performance_measures.items():
            self.scores.loc['score', name] = method(y_true, y_predict)
        ''' calculate group fairness measures '''
        for name, method in _group_fairness_measures.items():
            ''' Privileged group =  1 '''
            if name == 'group_theil_index':
                self.scores.loc['score', name] = method(y_true['class'], y_predict, alpha=1, pos_label=1)
            else:
                self.scores.loc['score', name] = abs(method(y_true, y_predict, prot_attr='group', priv_group=1))
        ''' calculate individual fairness measures '''
        for name, method in _individual_fairness_measures.items():
            self.scores.loc['score', name] = method(b)   
        ''' ratio and difference of performance measures '''
        for assist_name, assist_method in _assist_measures.items():
            for name, method in _performance_measures.items():
                try:
                    if assist_name == 'ratio':
                        # for all values to fall between 0 and 1
                        if assist_method(method, y_true, y_predict, prot_attr='group', priv_group=1) > 1:
                            self.scores.loc['score', assist_name + '_' + name] = 1/assist_method(method, y_true, y_predict,
                                                                                                 prot_attr='group', 
                                                                                                 priv_group=1)
                        else:
                            self.scores.loc['score', assist_name + '_' + name] = assist_method(method, y_true, y_predict,
                                                                                               prot_attr='group', priv_group=1)
                    else:
                        self.scores.loc['score', assist_name + '_' + name] = abs(assist_method(method, y_true, y_predict,
                                                                                               prot_attr='group', priv_group=1))
                except: # this is for when the stratification does not contemplate the tuple (group, class)
                    if assist_name == 'ratio':
                        self.scores.loc['score', assist_name + '_' + name] = 0
                    else:
                        self.scores.loc['score', assist_name + '_' + name] = 1
            
    def get_groupfairness_scores(self):
        return self.scores[get_group_fairness_measure_names()]
    
    def get_individualfairness_scores(self):
        return self.scores[get_individual_fairness_measure_names()]
    
    def get_performance_scores(self):
        return self.scores[get_performance_measure_names()]
    
    def get_ratio_scores(self):
        columns = list(map(lambda x: 'ratio_' + x, get_performance_measure_names()))
        return self.scores[columns]
    
    def get_difference_scores(self):
        columns = list(map(lambda x: 'dif_' + x, get_performance_measure_names()))
        return self.scores[columns]
    
    def calculate(self, y_true, y_predict, group_indexes, privileged_group):
        
        self.privileged_group = privileged_group
        
        ''' AIF360 format '''
        y_true = pd.DataFrame(y_true, columns=['class'])
        y_true.index = convert_index(list(group_indexes), self.privileged_group)
        y_true.index.names = ['group']
        self._initialize_reports()
        self._update_scores(y_true, y_predict)