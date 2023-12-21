import pandas as pd
import numpy as np
import os
import re

from sklearn.preprocessing import StandardScaler

''' Current directory ''' 
cdir = os.getcwd()
''' Target directory '''
path = os.path.join(os.path.abspath(os.path.join(cdir, os.pardir)), 'Original Datasets')

class DataSet:
    ''' Class for using fairness datasets '''  

    def __init__(self, print_runs=False):
        self.print_runs = print_runs
        self.protected_attributes = [None]
        self.dataset = pd.DataFrame()
        self.class_attribute = ''

    def get_protected_attributes(self):
        ''' Return protected attributes '''
        return self.protected_attributes

    def set_protected_attributes(self, protected_attributes):
        ''' Set protected attributes '''
        if not isinstance(protected_attributes, list):
            raise ValueError('protected_attributes values must be of type - list -')

        if not set(protected_attributes).issubset(set(self.dataset.columns)):
            raise ValueError('protected_attributes values are unknown in the dataset.')

        self.protected_attributes = protected_attributes
        self.generate_multiindex()

    def remove_rows_with_nan(self, symbol=np.nan):
        ''' Remove examples with null value '''
        n_rows = self.dataset.shape[0]
        self.dataset = self.dataset.replace(symbol, np.nan)
        self.dataset = self.dataset.dropna()
        if self.print_runs:
            print('Execute: remove_rows_with_nan\n---Removed %d rows in the table.'
                  % (n_rows - self.dataset.shape[0]))

    def remove_duplicates(self):
        ''' Remove duplicate examples '''
        n_rows = self.dataset.shape[0]
        ''' Keep one of the duplicate examples - keep = 'first' '''
        self.dataset = self.dataset.drop_duplicates(self.dataset.columns, keep='first')
        if self.print_runs:
            print('Execute: remove_duplicates\n---Removed %d rows in the table.'
                  % (n_rows - self.dataset.shape[0]))

    def remove_inconsistent_class(self):
        ''' Remove inconsistent classes '''
        n_rows = self.dataset.shape[0]
        columns = list(self.dataset.columns)
        columns.remove(self.class_attribute)
        self.dataset = self.dataset.drop_duplicates(subset=columns, keep=False)
        if self.print_runs:
            print('Execute: remove_inconsistent_class\n---Removed %d rows in the table.'
                  % (n_rows - self.dataset.shape[0]))

    def remove_protected_attributes(self):
        ''' Remove protected attributes '''
        self.dataset = self.dataset.drop(self.protected_attributes, axis=1)
        if self.print_runs:
            print('Execute:remove_protected_attributes\n---Removed %s in the table.'
                  % self.protected_attributes)

    def generate_multiindex(self):
        data_protected_attributes = self.dataset[self.protected_attributes]
        multindex = pd.MultiIndex.from_frame(data_protected_attributes)
        self.dataset = pd.DataFrame(self.dataset.to_numpy(), index=multindex, columns=self.dataset.columns)
        
    def get_groups(self):
        return list(np.unique(list(map('/'.join, list(self.dataset.index)))))

    def get_xy(self):
        ''' Split the dataset into X and y ''' 
        x = self.dataset.loc[:, self.dataset.columns != self.class_attribute]
        y = np.array(self.dataset.loc[:, self.dataset.columns == self.class_attribute]).ravel()
        return x, y

    def data_filtering(self, filter_dict):
        ''' Filter the dataset '''
        for column, values in filter_dict.items():
            self.dataset = self.dataset.loc[self.dataset[column].isin(values)]
                         
    def set_dataset(self, dataset):
        ''' Force use a preprocessed dataset '''
        self.dataset = dataset

''' Dataset Adult a.k.a Census Income '''
class Adult(DataSet):
    def __init__(self, print_runs=False):
        self.print_runs = print_runs
        self.dataset = self._open_dataset()
        self.class_attribute = 'class'
        self.protected_attributes = ['sex', 'race']
        self.generate_multiindex()
        self.dataset = self.dataset.astype(self._get_initial_columns_type())

    def _open_dataset(self):
        ''' Concatenate '''
        part1 = pd.read_csv(os.path.join(path, 'adult.data'), names=self._get_initial_columns_name())
        part2 = pd.read_csv(os.path.join(path, 'adult.test'), names=self._get_initial_columns_name())
        dataset = pd.concat([part1, part2])
        ''' Remove whitespace from strings '''
        dataset = dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        ''' Adjust the name of target classes '''
        dataset['class'] = dataset['class'].apply(self._adjust_class)
        ''' Adjust the relationship attribute values '''
        dataset['relationship'] = dataset['relationship'].apply(self._adjust_relationship)
        return dataset

    def basic_preprocessing(self):
        ''' Perform basic preprocessing for the dataset '''
        self.remove_protected_attributes()
        self.dataset = self.dataset.drop('education', axis=1)
        self.remove_rows_with_nan('?')
        self.remove_duplicates()
        self.remove_inconsistent_class()
        self.dataset['class'] = self.dataset['class'].apply(self._class_to_num).astype(int)
        self.dataset = pd.get_dummies(self.dataset, prefix_sep='=', drop_first=True, dtype=int)
        ''' Standardize features '''
        x, y = self.get_xy()
        standard_scale = StandardScaler().fit(x)
        self.dataset = pd.DataFrame(standard_scale.transform(x), columns=x.columns, index=x.index)
        self.dataset['class'] = y

    @staticmethod
    def _get_initial_columns_name():
            return ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',  'native-country',
                    'class']

    @staticmethod
    def _get_initial_columns_type():
        return {
            'age': int,
            'workclass': 'category',
            'fnlwgt': float,
            'education': 'category',
            'education-num': int,
            'marital-status': 'category',
            'occupation': 'category',
            'relationship': 'category',
            'race': 'category',
            'sex': 'category',
            'capital-gain': float,
            'capital-loss': float,
            'hours-per-week': float,
            'native-country': 'category',
            'class': 'category'
        }

    @staticmethod
    def _adjust_class(x):
        if x == '<=50K.':
            return '<=50K'
        elif x == '>50K.':
            return '>50K'
        else:
            return x

    @staticmethod
    def _class_to_num(x):
        if x == '>50K.' or x == '>50K':
            return 1
        else:
            return 0

    @staticmethod
    def _adjust_relationship(x):
        if x == 'Husband' or x == 'Wife':
            return 'Married'
        else:
            return x

    @staticmethod
    def _include_foreign(x):
        if x == 'United-States':
            return 0
        else:
            return 1
        
class Arrhythmia(DataSet):
    def __init__(self, print_runs=False):
        self.print_runs = print_runs
        self.dataset = self._open_dataset()
        self.class_attribute = 'class'
        self.protected_attributes = ['sex']
        self.generate_multiindex()

    def _open_dataset(self):
        dataset = pd.read_csv(os.path.join(path, 'arrhythmia.data'), index_col=False, names=self._get_initial_columns_name(),
                              sep=',')
        dataset = dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        dataset['sex'] = dataset['sex'].apply(self._att_sex).astype('category')
        ''' Binarize the target class '''
        dataset['class'] = dataset['class'].apply(self._att_class).astype('category')
        return dataset
    
    def basic_preprocessing(self):
        ''' Perform basic preprocessing for the dataset '''
        self.dataset = self.dataset.drop(['j'], axis=1)
        self.remove_rows_with_nan('?') 
        self.remove_duplicates()
        self.remove_inconsistent_class()
        self.dataset['sex'] = self.dataset['sex'].apply(self._att_sex_num).astype('int')
        self.dataset['class'] = self.dataset['class'].apply(self._att_class_num).astype('int')
        ''' Standardize features '''
        x, y = self.get_xy()
        standard_scale = StandardScaler().fit(x)
        self.dataset = pd.DataFrame(standard_scale.transform(x), columns=x.columns, index=x.index)
        self.dataset[self.class_attribute] = y
    
    @staticmethod
    def _get_initial_columns_name():
        columns = ['age', 'sex', 'height', 'weight', 'qrs_duration', 'pr_interval', 'qt_inteval', 't_interval', 
                   'p_interval', 'qrs','t','p', 'qrst', 'j', 'heart_rate', 'di_q', 'di_r', 'di_s', 'di_r2', 'di_s2', 
                   'intri_deflections', 'exi_ragged_r', 'exi_diphasic_r', 'exi_ragged_p', 'exi_diphasic_p', 
                   'exi_ragged_t', 'exi_diphasic_t', ]
        for value in range(28,280):
            columns.append('channel_' + str(value))   
        columns.append('class')
        
        return columns
            
    @staticmethod
    def _att_sex(x):
        if x == 0:
            return 'Male'
        else:
            return 'Female'
        
    @staticmethod
    def _att_class(x):
        if x == 1:
            return 'Normal'
        else: 
            return 'Arrhythmia'

    @staticmethod
    def _att_sex_num(x):
        if x == 'Male':
            return 0
        else:
            return 1
   
    @staticmethod
    def _att_class_num(x):
        if x == 'Arrhythmia':
            return 1
        else: 
            return 0

class Bank(DataSet):
    def __init__(self, print_runs=False):
        self.print_runs = print_runs
        self.dataset = self._open_dataset()
        self.class_attribute = 'y'
        self.protected_attributes = ['age']
        self.generate_multiindex()
        self.dataset = self.dataset.astype(self._get_initial_columns_type())

    def _open_dataset(self):
        dataset = pd.read_csv(os.path.join(path, 'bank-full.csv'), index_col=False, sep=';')
        dataset = dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        dataset['age'] = dataset['age'].apply(self._att_age).astype(str)
        bool_attributes = ['default', 'housing', 'loan', 'y']
        for attribute in bool_attributes:
            dataset[attribute] = dataset[attribute].apply(self._att_no_yes).astype(int)
        return dataset

    def basic_preprocessing(self):
        ''' Perform basic preprocessing for the dataset '''
        self.remove_rows_with_nan()
        self.remove_duplicates()
        self.remove_inconsistent_class()
        self.dataset = pd.get_dummies(self.dataset, prefix_sep='=', drop_first=True, dtype=int)
        ''' Standardize features '''
        types = self.dataset.dtypes.to_dict()
        x, y = self.get_xy()
        standard_scale = StandardScaler().fit(x)
        self.dataset = pd.DataFrame(standard_scale.transform(x), columns=x.columns, index=x.index)
        self.dataset[self.class_attribute] = y

    @staticmethod
    def _get_initial_columns_type():
        return {
            'age': 'category',
            'job': 'category',
            'marital': 'category',
            'education': 'category',
            'default': int,
            'balance': float,
            'housing': int,
            'loan': int,
            'contact': 'category',
            'day': int,
            'month': 'category',
            'duration': int,
            'campaign': int,
            'pdays': int,
            'previous': int,
            'poutcome': 'category',
            'y': int
        }

    @staticmethod
    def _att_age(x):
        if x > 25:
            return '>25'
        else:
            return '<=25'

    @staticmethod
    def _att_no_yes(x):
        if x == 'yes':
            return 1
        else:
            return 0

class Compas(DataSet):
    def __init__(self, print_runs=False):
        self.print_runs = print_runs
        self.dataset = self._open_dataset()
        self.class_attribute = 'two_year_recid'
        self.protected_attributes = ['sex', 'race']
        self.generate_multiindex()

    def _open_dataset(self):
        dataset = pd.read_csv(os.path.join(path, 'compas-scores-two-years.csv'), index_col=False, header=0, sep=',')
        dataset = dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        return dataset
    
    def basic_preprocessing(self):
        ''' Perform basic preprocessing for the dataset '''
        self.remove_protected_attributes()
        self.dataset = self.dataset.drop(['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'age_cat', 
                                          'c_case_number', 'c_jail_in', 'c_jail_out', 'c_offense_date', 'c_arrest_date', 
                                          'r_case_number', 'r_offense_date', 'r_jail_in', 'r_jail_out', 'vr_offense_date', 
                                          'screening_date', 'v_screening_date', 'type_of_assessment', 'v_type_of_assessment', 
                                          'in_custody', 'out_custody', 'event', 'days_b_screening_arrest', 'start', 'end',
                                          'priors_count.1', 'v_decile_score', 'v_score_text', 'is_recid', 'violent_recid',
                                          'r_charge_degree', 'r_days_from_arrest', 'r_charge_desc', 'is_violent_recid',
                                          'vr_case_number', 'vr_charge_degree', 'vr_charge_desc', 'decile_score.1'
                                          ], axis = 1)
        self.dataset['c_charge_desc'] = self.dataset['c_charge_desc'].str.upper()
        self.remove_rows_with_nan()
        self.remove_duplicates()
        self.remove_inconsistent_class()
        self.dataset = pd.get_dummies(self.dataset, prefix_sep='=', drop_first=True, dtype=int)
        ''' Standardize features '''
        x, y = self.get_xy()
        standard_scale = StandardScaler().fit(x)
        self.dataset = pd.DataFrame(standard_scale.transform(x), columns=x.columns, index=x.index)
        self.dataset[self.class_attribute] = y
        ''' Reorganize column names to avoid problems with XGBoost '''
        columns_update = []
        for name in self.dataset.columns:
            columns_update.append(re.sub(r"[^a-zA-Z0-9-=]","_", name))
        self.dataset.columns = columns_update        
    
    @staticmethod
    def _get_initial_columns_name():
        return ['id', 'name', 'first', 'last', 'compas_screening_date', 'sex', 'dob', 'age', 'age_cat', 'race', 
                'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 
                'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date', 
                'c_days_from_compas', 'c_charge_degree', 'c_charge_desc', 'is_recid', 'r_case_number', 'r_charge_degree', 
                'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out', 'violent_recid', 
                'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 
                'type_of_assessment', 'decile_score.1', 'score_text', 'screening_date', 'v_type_of_assessment', 
                'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody', 'out_custody', 'priors_count.1', 'start',
                'end', 'event', 'two_year_recid']
    
    @staticmethod
    def _get_initial_columns_type():
        return {
            'id' : int, # delete
            'name' : object, # delete
            'first' : object, # delete
            'last' : object, # delete
            'compas_screening_date' : object, # date (delete)
            'sex' : 'category', 
            'dob' : object, # date (delete)
            'age' : int, 
            'age_cat' : 'category',
            'race' : 'category',
            'juv_fel_count' : int, 
            'decile_score' : 'category',
            'juv_misd_count' : int, 
            'juv_other_count' : int,
            'priors_count' : int,
            'days_b_screening_arrest' : int,
            'c_jail_in' : object, # date (delete)
            'c_jail_out' : object, # date (delete)
            'c_case_number' : object, # delete
            'c_offense_date' : object, # date (delete)
            'c_arrest_date' : object, # date (delete)
            'c_days_from_compas' : int, 
            'c_charge_degree' : 'category',
            'c_charge_desc' : object, # delete
            'is_recid' : 'category', # delete
            'r_case_number' : object, # delete
            'r_charge_degree' : 'category',
            'r_days_from_arrest' : int, # delete
            'r_offense_date' : object, # date (delete)
            'r_charge_desc' : object, # description (delete)
            'r_jail_in' : object, # date (delete)
            'r_jail_out' : object, # date (delete)
            'violent_recid' : 'category', # delete
            'is_violent_recid' : 'category', # delete
            'vr_case_number' : object, # delete
            'vr_charge_degree' : 'category', 
            'vr_offense_date' : object, # date (delete)
            'vr_charge_desc' : object, # description (deletar)
            'type_of_assessment' : object, # delete 
            'decile_score.1' : int, # delete
            'score_text' : 'category', # delete 
            'screening_date' : object, # date (delete)
            'v_type_of_assessment' : object, # delete
            'v_decile_score' : int, # delete
            'v_score_text' : 'category', # delete
            'v_screening_date' : object, # date (delete)
            'in_custody' : object, # date (delete)
            'out_custody' : object, # date (delete)
            'priors_count.1' : int, # delete
            'start' : int, # delete
            'end' : int, # delete
            'event' : 'category',
            'two_year_recid' : int
        }
    
class Contraceptive(DataSet):
    def __init__(self, print_runs=False):
        self.print_runs = print_runs
        self.class_attribute = 'use_contraceptive'
        self.protected_attributes = ['religion']
        self.dataset = self._open_dataset()
        self.generate_multiindex()
        
    def _open_dataset(self):
        dataset = pd.read_csv(os.path.join(path, 'cmc.data'), header=None, index_col=False, sep=',', 
                              names=self._get_initial_columns_name())
        dataset['religion'] = dataset['religion'].apply(self._att_religion)
        ''' Binarize the target class '''
        dataset['use_contraceptive'] = dataset['use_contraceptive'].apply(self._att_contraceptive)
        return dataset
        
    def basic_preprocessing(self):
        ''' Perform basic preprocessing for the dataset '''
        self.remove_protected_attributes()
        self.remove_rows_with_nan('?')
        self.remove_duplicates()
        self.remove_inconsistent_class()
        self.dataset['age'] = self.dataset['age'].astype('int')
        self.dataset['education'] = self.dataset['education'].astype('int')
        self.dataset['husband_education'] = self.dataset['husband_education'].astype('int')
        self.dataset['children'] = self.dataset['children'].astype('int')
        self.dataset['working'] = self.dataset['working'].astype('category')
        self.dataset['husband_occupation'] = self.dataset['husband_occupation'].astype('category')
        self.dataset['standard_living'] = self.dataset['standard_living'].astype('int')
        self.dataset['media_exposure'] = self.dataset['media_exposure'].astype('category')
        x, y = self.get_xy()
        x = pd.get_dummies(x, prefix_sep='=', drop_first=True, dtype=int)
        ''' Standardize features '''
        standard_scale = StandardScaler().fit(x)
        self.dataset = pd.DataFrame(standard_scale.transform(x), columns=x.columns, index=x.index)
        self.dataset[self.class_attribute] = y
    
    @staticmethod
    def _get_initial_columns_name():
        return ('age', 'education', 'husband_education', 'children', 'religion', 'working', 'husband_occupation', 
                'standard_living', 'media_exposure', 'use_contraceptive')
    
    @staticmethod
    def _att_religion(x):
        if x == 0:
            return 'Non-Islam'
        else:
            return 'Islam'
        
    @staticmethod
    def _att_contraceptive(x):
        if x == 1:
            return 0
        else:
            return 1
        
class Drug(DataSet):
    ''' UCI https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29 '''
    def __init__(self, class_attribute, print_runs=False):
        self.print_runs = print_runs
        self.class_attribute = class_attribute
        self.protected_attributes = ['ethnicity']
        self.dataset = self._open_dataset()
        self.generate_multiindex()

    def _open_dataset(self):
        dataset = pd.read_csv(os.path.join(path, 'drug_consumption.data'), index_col=0, sep=',', 
                              names=self._get_initial_columns_name())
        dataset = dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        drop = ['alcohol', 'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack', 
                'ecstasy', 'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 'mushrooms', 'nicotine', 'semer', 'vsa']
        drop.remove(self.class_attribute)
        dataset = dataset.drop(drop, axis=1)
        dataset['ethnicity'] = dataset['ethnicity'].apply(self._att_ethnicity).astype('category')
        dataset[self.class_attribute] = dataset[self.class_attribute].apply(self._att_class).astype('category')
        return dataset
    
    def basic_preprocessing(self):
        ''' Perform basic preprocessing for the dataset '''
        self.remove_protected_attributes()
        self.dataset[self.class_attribute] = self.dataset[self.class_attribute].apply(self._att_class_num).astype('category')
        self.remove_rows_with_nan()
        self.remove_duplicates()
        self.remove_inconsistent_class()
        self.dataset = self.dataset.astype('float')
        self.dataset[self.class_attribute] = self.dataset[self.class_attribute].astype('int') 
            
    @staticmethod
    def _get_initial_columns_name():
        return ['age', 'gender', 'education', 'country', 'ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 
                'impulsive', 'ss', 'alcohol', 'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack', 
                'ecstasy', 'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 'mushrooms', 'nicotine', 'semer', 'vsa']
      
    @staticmethod
    def _att_ethnicity(x):
        if x == -0.31685:
            return 'White'
        else:
            return 'Non-White'
    
    @staticmethod
    def _att_class(x):
        if x == 'CL0':
            return 'Never-Used'
        else:
            return 'Others'
        
    @staticmethod
    def _att_class_num(x):
        if x == 'Never-Used':
            return 0
        else:
            return 1       

class DrugCrack(Drug, DataSet):
    def __init__(self, print_runs=False):
        super().__init__('crack', print_runs)
        
class DrugHeroin(Drug, DataSet):
    def __init__(self, print_runs=False):
        super().__init__('heroin', print_runs)

class German(DataSet):
    def __init__(self, print_runs=False):
        self.print_runs = print_runs
        self.dataset = self._open_dataset()
        self.class_attribute = 'class'
        self.protected_attributes = ['sex']
        self.generate_multiindex()
        self.dataset = self.dataset.astype(self._get_initial_columns_type())

    def _open_dataset(self):
        dataset = pd.read_csv(os.path.join(path, 'german.data'), index_col=False, names=self._get_initial_columns_name(),
                              sep=' ')
        dataset = dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        dataset['sex'] = dataset['sex'].apply(self._att_sex).astype(str)
        dataset['class'] = dataset['class'].apply(self._att_class).astype(int)
        dataset['status'] = dataset['status'].apply(self._att_status).astype(int)
        dataset['savings'] = dataset['savings'].apply(self._att_savings).astype(int)
        dataset['employment since'] = dataset['employment since'].apply(self._att_employment).astype(int)
        dataset['job'] = dataset['job'].apply(self._att_job).astype(int)
        return dataset

    def basic_preprocessing(self):
        ''' Perform basic preprocessing for the dataset '''
        self.remove_protected_attributes
        self.remove_rows_with_nan()
        self.remove_duplicates()
        self.remove_inconsistent_class()
        self.dataset = pd.get_dummies(self.dataset, prefix_sep='=', drop_first=True, dtype=int)
        ''' Standardize features '''
        x, y = self.get_xy()
        standard_scale = StandardScaler().fit(x)
        self.dataset = pd.DataFrame(standard_scale.transform(x), columns=x.columns, index=x.index)
        self.dataset[self.class_attribute] = y

    @staticmethod
    def _get_initial_columns_name():
        return ['status', 'duration (month)', 'credit history', 'purpose', 'credit amount', 'savings',
                'employment since', 'installment rate', 'sex', 'other debtors', 'residence since',
                'property', 'age', 'installment plans', 'housing', 'credits', 'job', 'peoples',
                'phone', 'foreign worker', 'class']

    @staticmethod
    def _get_initial_columns_type():
        return {
            'status': int,
            'duration (month)': int,
            'credit history': 'category',
            'purpose': 'category',
            'credit amount': float,
            'savings': int,
            'employment since': int,
            'installment rate': float,
            'sex': 'category',
            'other debtors': 'category',
            'residence since': int,
            'property': 'category',
            'age': int,
            'installment plans': 'category',
            'housing': 'category',
            'credits': int,
            'job': int,
            'peoples': int,
            'phone': 'category',
            'foreign worker': 'category',
            'class': int
        }

    @staticmethod
    def _att_sex(x):
        if x == 'A91' or x == 'A93' or x == 'A94':
            return 'Male'
        else:
            return 'Female'

    @staticmethod
    def _att_class(x):
        if x == 2:
            return 0
        else:
            return 1

    @staticmethod
    def _att_status(x):
        # < 0 DM
        if x == 'A11':
            return 0
        # 0 <= ... < 200 DM
        elif x == 'A12':
            return 2
        # ... >= 200 DM / salary assignments for at least 1 year
        elif x == 'A13':
            return 3
        # no checking account
        else:
            return 1

    @staticmethod
    def _att_savings(x):
        # ... < 100 DM
        if x == 'A61':
            return 1
        # 100 <= ... < 500 DM
        elif x == 'A62':
            return 2
        # 500 <= ... < 1000 DM
        elif x == 'A63':
            return 3
        # .. >= 1000 DM
        elif x == 'A64':
            return 4
        # unknown/ no savings account
        else:
            return 0

    @staticmethod
    def _att_employment(x):
        # unemployed
        if x == 'A71':
            return 0
        # ... < 1 year
        elif x == 'A72':
            return 1
        # 1 <= ... < 4 years
        elif x == 'A73':
            return 2
        # 4 <= ... < 7 years
        elif x == 'A74':
            return 3
        # .. >= 7 years
        else:
            return 4

    @staticmethod
    def _att_job(x):
        # unemployed/ unskilled - non-resident
        if x == 'A171':
            return 0
        # unskilled - resident
        elif x == 'A172':
            return 1
        # skilled employee / official
        elif x == 'A173':
            return 2
        # management/ self-employed/highly qualified employee/ officer
        else:
            return 3

class Heart(DataSet):
    def __init__(self, print_runs=False):
        self.print_runs = print_runs
        self.class_attribute = 'class'
        self.protected_attributes = ['age']
        self.dataset = self._open_dataset()
        self.generate_multiindex()
   
    def _open_dataset(self):
        dataset = pd.read_csv(os.path.join(path, 'processed.cleveland.data'), index_col=False, sep=',', 
                              names=self._get_initial_columns_name())
        dataset = dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        dataset['age'] = dataset['age_num'].apply(self._att_age).astype('category')
        dataset['class'] = dataset['class'].apply(self._att_class)
        return dataset
    
    def basic_preprocessing(self):
        ''' Perform basic preprocessing for the dataset '''
        self.remove_protected_attributes()
        self.remove_rows_with_nan('?')
        self.remove_duplicates()
        self.remove_inconsistent_class()
        self.dataset = self.dataset.astype('float')
        self.dataset[self.class_attribute] = self.dataset[self.class_attribute].astype('int') 
        ''' Standardize features '''
        x, y = self.get_xy()
        standard_scale = StandardScaler().fit(x)
        self.dataset = pd.DataFrame(standard_scale.transform(x), columns=x.columns, index=x.index)
        self.dataset[self.class_attribute] = y
    
    @staticmethod
    def _get_initial_columns_name():
        columns = ['age_num', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                   'slope', 'ca','thal','class']
        return columns
        
    @staticmethod
    def _att_age(x):
        if x >= 58:
            return 'Senior'
        else:
            return 'Non-Senior'  

    @staticmethod
    def _att_class(x):
        if x == 0:
            return 0
        else:
            return 1

class Student(DataSet):
    
    def __init__(self, print_runs=False):
        self.print_runs = print_runs
        self.class_attribute = 'Class'
        self.protected_attributes = ['gender']
        self.dataset = self._open_dataset()
        self.generate_multiindex()
        
    def _open_dataset(self):
        dataset = pd.read_csv(os.path.join(path, 'xAPI-Edu-Data.csv'), header=0, index_col=False, sep=',')
        ''' Reajusta nome da nacionalidade e deleta colunas com informações redundantes '''
        #dataset['Nationality'] = list(dataset['NationalITy'])
        dataset = dataset.rename(columns = {'NationalITy':'Nationality'})
        #dataset['Nationality'] = dataset['Nationality'].apply(self._att_nationality)
        dataset = dataset.drop(columns = ['PlaceofBirth'])
        ''' Redefine a classe entre quem não e quem usa método contraceptivo (short e long term) '''
        dataset['Class'] = dataset['Class'].apply(self._att_class)
        return dataset
    
    def basic_preprocessing(self):
        ''' Deleta o atributo protegido '''
        self.remove_protected_attributes()
        ''' Realiza as remoções com exemplos com dados faltantes, duplicados ou com classe inconsistente '''
        self.remove_rows_with_nan('?')
        self.remove_duplicates()
        self.remove_inconsistent_class()
        ''' Ajusta as colunas '''
        self.dataset = self.dataset.replace({'StageID' : self._att_stageID()})
        self.dataset['raisedhands'] = self.dataset['raisedhands'].astype('int')
        self.dataset['VisITedResources'] = self.dataset['VisITedResources'].astype('int')
        self.dataset['AnnouncementsView'] = self.dataset['AnnouncementsView'].astype('int')
        self.dataset['Discussion'] = self.dataset['Discussion'].astype('int')
        ''' Transforma os atributos categóricos '''
        x, y = self.get_xy()
        x = pd.get_dummies(x, prefix_sep='=', drop_first=True, dtype=int)
        ''' Normaliza os dados '''
        standard_scale = StandardScaler().fit(x)
        self.dataset = pd.DataFrame(standard_scale.transform(x), columns=x.columns, index=x.index)
        self.dataset[self.class_attribute] = y
    
    @staticmethod
    def _att_stageID():
        return {'lowerlevel' : 0, 'MiddleSchool' : 1, 'HighSchool' : 2}
    
    @staticmethod
    def _att_class(x):
        if x == 'L':
            return 0
        else: 
            return 1
        
class Titanic(DataSet):
    def __init__(self, print_runs=False):
        self.print_runs = print_runs
        self.dataset = self._open_dataset()
        self.class_attribute = 'Survived'
        self.protected_attributes = ['Sex']
        self.generate_multiindex()
        self.dataset = self.dataset.astype(self._get_initial_columns_type())

    def _open_dataset(self):
        dataset = pd.read_csv(os.path.join(path, 'titanic.csv'),  index_col=False, sep=',')
        dataset = dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        dataset = dataset.drop('Name', axis=1)
        return dataset

    def basic_preprocessing(self):
        ''' Perform basic preprocessing for the dataset '''
        self.remove_rows_with_nan()
        self.remove_duplicates()
        self.remove_inconsistent_class()
        self.dataset = pd.get_dummies(self.dataset, prefix_sep='=', drop_first=True, dtype=int)
        ''' Standardize features '''
        x, y = self.get_xy()
        standard_scale = StandardScaler().fit(x)
        self.dataset = pd.DataFrame(standard_scale.transform(x), columns=x.columns, index=x.index)
        self.dataset[self.class_attribute] = y

    @staticmethod
    def _get_initial_columns_type():
        return {
            'Survived': int,
            'Pclass': int,
            'Sex': 'category',
            'Age': int,
            'Siblings/Spouses Aboard': int,
            'Parents/Children Aboard': int,
            'Fare': float
        }