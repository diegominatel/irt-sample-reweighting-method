import pandas as pd
import numpy as np
import os

''' Current directory ''' 
cdir = os.path.abspath(os.path.join(__file__, os.pardir))

class Dataset:
    ''' Class responsible for loading preprocessed datasets '''
    def __init__(self):
        self.dataset = pd.DataFrame()
      
    def get_xy(self):
        ''' Split the dataset into X and y ''' 
        x = self.dataset.loc[:, self.dataset.columns != 'target']
        y = np.array(self.dataset.loc[:, self.dataset.columns == 'target']).ravel()
        return x, y

class Adult(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'adult.csv'), sep=';', index_col='Group')
    
class Arrhythmia(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'arrhythmia.csv'), sep=';', index_col='Group')
        
class Bank(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'bank.csv'), sep=';', index_col='Group')
        
class CompasMen(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'compasMen.csv'), sep=';', index_col='Group')

class CompasWomen(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'compasWomen.csv'), sep=';', index_col='Group')
        
class Contraceptive(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'contraceptive.csv'), sep=';', index_col='Group')
        
class Crack(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'crack.csv'), sep=';', index_col='Group')
        
class German(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'german.csv'), sep=';', index_col='Group')
        
class Heart(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'heart.csv'), sep=';', index_col='Group')
        
class Heroin(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'heroin.csv'), sep=';', index_col='Group')
        
class Student(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'student.csv'), sep=';', index_col='Group')
        
class Titanic(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(cdir, 'titanic.csv'), sep=';', index_col='Group')
        