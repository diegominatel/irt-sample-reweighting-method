# -*- coding: utf-8 -*-

''' General packages '''
import pandas as pd

''' Load the methods to use R '''
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

''' Load the 'mirt' package in R '''
mirt_package = importr('mirt')
mirt_package.mirtCluster()

class Mirt:
    '''
    Class that runs the mirt package in R
    '''
    
    def __init__(self):
        self.__model = None
        self.params = None
        self.estimated_abilities = None
        
    def mirt_config(self, group):
        '''
        Returns the parameters used in mirt
        '''
        technical = ro.r('list(NCYCLES = 500, message=FALSE, warn=FALSE)') 
        params = dict() 
        params['technical'] = technical
        params['itemtype'] = '2PL'
        params['group'] = group
        params['verbose'] = True
        params['invariance'] = ro.vectors.StrVector(['intercepts'])
        
        return params
    
    def calculate(self, item_modeling, group):
        '''
        Args:
                item_modeling:  (pd.DataFrame) where rows indicate individuals and columns indicate items, each value pair i,j 
                                represents whether individual i correctly answered item j.
                group:   (list) indicates the group ('Privileged' or 'Unprivileged') to which the individual belongs.
        '''
        item_names = list(item_modeling.columns)
        ''' Convert DataFrame to R Frame '''
        with localconverter(ro.default_converter + pandas2ri.converter):
            item_modeling = ro.conversion.py2rpy(item_modeling)
        ''' Convert List to R Vector '''    
        group = ro.vectors.StrVector(group)
        ''' Run mirt '''
        self.__model = mirt_package.multipleGroup(item_modeling, 1, **self.mirt_config(group))
        ''' Estimated abilities '''
        ability = mirt_package.fscores(self.__model)
        ''' Set the attributes of the class (Convert R Frame to Pandas) '''
        with localconverter(ro.default_converter + pandas2ri.converter):
            self.estimated_abilities = pd.DataFrame(ro.conversion.rpy2py(ability), columns=['ability'])