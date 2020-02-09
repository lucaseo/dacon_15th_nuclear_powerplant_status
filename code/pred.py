import os
import pandas as pd 
import numpy as np
from multiprocessing import Pool 
import multiprocessing
from tqdm import tqdm
from functools import partial
import datetime

import sys
sys.path.append('source/')
from data_loader_v2 import data_loader_v2
import lightgbm as lgb

model_name = str(sys.argv[1])
submission_file_name = str(sys.argv[2])

EVENT_TIME = 8

test_path = '../../../datadrive/plant/test/'
test_list = os.listdir(test_path)

print('Loading test data ...')
def data_loader_all_v2(func, files, folder='', train_label=None, event_time=10, nrows=60):   
    func_fixed = partial(func, folder=folder, train_label=train_label, event_time=event_time, nrows=nrows)     
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    return combined_df
test = data_loader_all_v2(data_loader_v2, test_list, folder=test_path, train_label=None, event_time=EVENT_TIME, nrows=60)


print('Start data preprocessing ...')
with open('filter_col.txt', 'r') as filehandle:
    list_ = filehandle.readlines()
list_ = [col.replace('\n', '') for col in list_]
test = test[list_]
for col in test.columns:
    if test[col].dtype != 'float64':
        test[col] = pd.to_numeric(test[col], errors='coerce')
test = test.replace(np.nan, 0, regex=True)
test = test.fillna(value=0)


print('Start prediction')
if model_name == 'lgb':
    
    bst = lgb.Booster(model_file='model_lgb.txt')
    pred = bst.predict(test, num_iteration=bst.best_iteration)

elif model_name == 'rf':
    rf_clf = pickle.load(open('model_rf.pickle', 'rb'))
    pred = rf_clf.predict_proba(test)


submission = pd.DataFrame(data=pred)
submission.index = test.index
submission.index.name = 'id'
submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission = submission.reset_index()

submission.to_csv('submission/' + str(submission_file_name) + '-' + str(datetime.datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss")) +'.csv', index=False)
print('Submission file created!')
