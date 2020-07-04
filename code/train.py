import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import warnings 
warnings.filterwarnings("ignore")

import os
import pandas as pd 
import numpy as np
from multiprocessing import Pool 
import multiprocessing
from tqdm import tqdm
from functools import partial

import sys
sys.path.append('../utils/')
from data_loader import data_loader


def data_loader_all(func, path, train, nrows, **kwargs):
    '''
    Parameters:
    
    func: 하나의 csv파일을 읽는 함수 
    path: [str] train용 또는 test용 csv 파일들이 저장되어 있는 폴더 
    train: [boolean] train용 파일들 불러올 시 True, 아니면 False
    nrows: [int] csv 파일에서 불러올 상위 n개의 row 
    lookup_table: [pd.DataFrame] train_label.csv 파일을 저장한 변수 
    event_time: [int] 상태_B 발생 시간 
    normal: [int] 상태_A의 라벨
    
    Return:
    
    combined_df: 병합된 train 또는 test data
    '''
    
    # 읽어올 파일들만 경로 저장 해놓기 
    files_in_dir = os.listdir(path)
    
    files_path = [path+'/'+file for file in files_in_dir]
    
    if train :
        func_fixed = partial(func, nrows = nrows, train = True, lookup_table = kwargs['lookup_table'], event_time = kwargs['event_time'], normal = kwargs['normal'])
        
    else : 
        func_fixed = partial(func, nrows = nrows, train = False)
    
    
    # 여러개의 코어를 활용하여 데이터 읽기 
    if __name__ == '__main__':
        pool = Pool(processes = multiprocessing.cpu_count()) 
        df_list = list(tqdm(pool.imap(func_fixed, files_path), total = len(files_path)))
        pool.close()
        pool.join()
    
    # 데이터 병합하기 
    combined_df = pd.concat(df_list, ignore_index=True)    
    return combined_df


train_path = '../../../datadrive/plant/train/'
test_path = '../../../datadrive/plant/test'
label = pd.read_csv('../../../datadrive/plant/train_label.csv')
EVENT_TIME = 0

## Load Train data
train = data_loader_all(data_loader, path=train_path, train = True, nrows = 600, normal = 999, event_time = EVENT_TIME, lookup_table = label)


## Preprocess
#1
train = train[train['label']!=999].reset_index(drop=True)
train_label = train.label
train = train.drop(['id','time','label'], axis=1)

#2
with open('filter_col.txt', 'r') as filehandle:
    list_ = filehandle.readlines()
list_ = [col.replace('\n', '') for col in list_]
train= train[list_]

# 3
for col in train.columns:
    if train[col].dtype != 'float64':
        train[col] = pd.to_numeric(train[col], errors='coerce')

# 4
train = train.fillna(value=0, axis=1)



## Train Val split
print('Start training ...')
X_train, X_valid = train_test_split(train, test_size = .25, random_state=42)
y_train, y_valid = train_test_split(train_label, test_size = .25, random_state=42)
X_train = X_train.to_numpy()
X_valid = X_valid.to_numpy()
y_train = y_train.to_numpy()
y_valid = y_valid.to_numpy()


## Train LightGBM
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train) #, feature_name=X_train.columns)
valid_data = lgb.Dataset(X_valid, label=y_valid) #, feature_name=X_valid.columns)
param = {
    'objective': 'multiclass',
    'num_class': 198,
    'boosting':'gbdt',  
    'num_leaves':32,
    'max_depth':20,
    'min_data_in_leaf':20,
    'metric':'multi_logloss', 
    'learning_rate' : 0.01,
    'num_threads' : 12,
    'verbose' : -1,
    'bagging_freq' : 1,
    'bagging_fraction' : 0.5,
    'feature_fraction' : 0.5,
}
evals_result={} 
num_round = 2000
lgbst = lgb.train(params=param, 
                train_set=train_data, 
                num_boost_round=num_round, 
                valid_sets=[valid_data], 
                evals_result=evals_result, 
                early_stopping_rounds=1000, 
                verbose_eval=10)
lgbst.save_model('model_lgb.txt', num_iteration=lgbst.best_iteration)