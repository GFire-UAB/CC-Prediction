#!/usr/bin/env python
# coding: utf-8
'''
Script used for performing Feature Selection on a given set of blocks with a given target variable (firstly used on CC).

Alejandro Donaire, Eric Sanchez, Pau Ventura, Francesco Tedesco
'''

## IMPORTS
# Data Handling
import pandas as pd
import numpy as np

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns
from xgboost import plot_importance

# Processing data
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

# Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Models
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Model tuning
from sklearn.model_selection import KFold, StratifiedKFold

# Others / utils
import time
import csv
import joblib
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from multiprocessing import Process, Queue
from os import listdir
from os.path import isfile, join
import psutil
import pprint

# Own utils
from ML_utils.reading_laz import *
from ML_utils.block_preprocessing_utils import *
from ML_utils.block_preprocessing_template import *
from ML_utils.block_preprocessing_merge import *
from ML_utils.LandUse_preprocessing import *

import logging
logging.basicConfig(level=logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)


## Constants
n_cpu = 2
groundtruth_precision = 20          
outliers = 135
total_processed_blocks = n_cpu*1 # Must be multiple of n_cpu
step = total_processed_blocks//n_cpu

data_path = '/lex/shared/projects/nnveget/data/laz'
control_test_path = '/lex/shared/projects/nnveget/data/control_test_laz'
tif_path = '/lex/shared/projects/nnveget/data/tifs'
landUse_datapath = join(tif_path, 'LandUseEsriCatRep.tif')
groundtruth_datapath = join(tif_path,'variables-biofisiques-arbrat-v1r0-cc-2016-2017.tif')
aspect_datapath = join(tif_path,'aspect.tif')
slope_datapath = join(tif_path,'slope.tif')

seasons_encoding = {'winter':0, 'spring':1, 'summer':2, 'autumn':3}
train_perc = 0.8

results_path = join('/lex/shared/projects/nnveget/results', str(total_processed_blocks))

print("Ram usage (starting point):", psutil.virtual_memory()[2])


## Metrics of the execution (time)
metrics = {}; metrics['n_blocks'] = total_processed_blocks; metrics['n_cpu'] = n_cpu 


## Stochastic block sampling
all_blockfiles = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
control_test_blockfiles = [join(control_test_path, f) for f in listdir(control_test_path) if isfile(join(control_test_path, f))]
blockfiles = np.array(list(set(all_blockfiles) - set(control_test_blockfiles)))

assert total_processed_blocks % n_cpu == 0, "ERROR: Processed blocks must be a multiple of ("+str(n_cpu)+")"
print("Processing:", total_processed_blocks, "blocks")

sampling = np.random.choice(len(blockfiles), len(blockfiles), replace=False) # Ensures replication
block_batches = sampling[:total_processed_blocks].reshape(n_cpu, step)

print("Block sampling done")
print("Blocks used:", block_batches)
print("Ram usage (after block sampling):", psutil.virtual_memory()[2])


## PARALLEL block preprocessing
TIME_start = time.time()

queue = Queue()
  
processes = [Process(target=preprocess_blocks_parallel, args=(blockfiles[batch], groundtruth_precision, queue, outliers,
                                                              seasons_encoding))
             for batch in block_batches]

for p in processes:
    p.start()

df_list = []
for p in processes:
    df_list.extend(queue.get())

for p in processes:
    p.join()


print("Ram usage (block preprocessing done):", psutil.virtual_memory()[2])

TIME_end = time.time()
metrics['parallel_preprocessing'] = TIME_end - TIME_start


## Reading other tiff info
TIME_start = time.time()

groundtruth = get_tif_info(groundtruth_datapath, 'CC')
groundtruth['x_p'] = groundtruth['x_p'].astype(np.int32)
groundtruth['y_p'] = groundtruth['y_p'].astype(np.int32)
gt_filter = groundtruth['CC'] > 100
groundtruth['CC'] = gt_filter * 100 + (~gt_filter) * groundtruth['CC'] # Fast vectorized implementation
	

TIME_end = time.time()
metrics['reading_tifs'] = TIME_end - TIME_start

print("Ram usage (after reading gt, slope and aspect):", psutil.virtual_memory()[2])
print("Reading tiff done")


## Merging in parallel
TIME_start = time.time()
queue = Queue()

processes = [Process(target=parallel_merging_tiffs, args=(df_list[i:i+step], groundtruth, queue))
             for i in np.arange(0, len(df_list), step)]

for p in processes:
    p.start()

processed_blocks = []
for p in processes:
    processed_blocks.extend(queue.get())

for p in processes:
    p.join()

del groundtruth
print("Ram usage (after merging gt):", psutil.virtual_memory()[2])
TIME_end = time.time()
metrics['merging tiffs'] = TIME_end - TIME_start

df_list = processed_blocks
del processed_blocks

slope = process_coords_aspect_slope( get_tif_info(slope_datapath, 'slope'), slope_datapath )
slope['x_p'] = slope['x_p'].astype(np.int32)
slope['x_p'] = slope['x_p'].astype(np.int32)


## Merging in parallel
TIME_start = time.time()
queue = Queue()

processes = [Process(target=parallel_merging_tiffs, args=(df_list[i:i+step], slope, queue))
             for i in np.arange(0, len(df_list), step)]

for p in processes:
    p.start()

processed_blocks = []
for p in processes:
    processed_blocks.extend(queue.get())

for p in processes:
    p.join()

del slope
print("Ram usage (after merging slope):", psutil.virtual_memory()[2])
TIME_end = time.time()
metrics['merging slope'] = TIME_end - TIME_start

df_list = processed_blocks
del processed_blocks

aspect = process_coords_aspect_slope( get_tif_info(aspect_datapath, 'aspect'), aspect_datapath )
aspect['x_p'] = aspect['x_p'].astype(np.int32)
aspect['x_p'] = aspect['x_p'].astype(np.int32)


## Merging in parallel
TIME_start = time.time()
queue = Queue()

processes = [Process(target=parallel_merging_tiffs, args=(df_list[i:i+step], aspect, queue))
             for i in np.arange(0, len(df_list), step)]

for p in processes:
    p.start()

processed_blocks = []
for p in processes:
    processed_blocks.extend(queue.get())

for p in processes:
    p.join()

del aspect 
print("Ram usage (after merging aspect):", psutil.virtual_memory()[2])
TIME_end = time.time()
metrics['merging aspect'] = TIME_end - TIME_start

print([x.shape for x in processed_blocks])
print([x.isna().sum()['CC'] for x in processed_blocks])


## Correcting NaN
for i in range(len(processed_blocks)):
    processed_blocks[i][['aspect', 'slope']] = processed_blocks[i][['aspect', 'slope']].replace(-9999, np.nan) # Will be imputed
    processed_blocks[i][['CC']] = processed_blocks[i][['CC']].fillna(0)


## Splitting data
TIME_start = time.time()

X_train = processed_blocks[:int(len(processed_blocks)*train_perc)]
X_test = processed_blocks[int(len(processed_blocks)*train_perc):]

train_blocks_coords = pd.concat([df.iloc[0][['x_p','y_p']] for df in X_train], axis=0).reset_index(drop=True)
test_blocks_coords = pd.concat([df.iloc[0][['x_p','y_p']] for df in X_test], axis=0).reset_index(drop=True)

X_train_df = pd.concat(X_train, axis=0).reset_index(drop=True); X_test_df = pd.concat(X_test, axis=0).reset_index(drop=True)

y_train = X_train_df['CC']; y_test = X_test_df['CC']
X_train_df.drop('CC', axis=1, inplace=True); X_test_df.drop('CC', axis=1, inplace=True)

train_seasons = X_train_df['season'].replace(0, 'winter'); test_seasons = X_test_df['season'].replace(0, 'winter')
X_train_df.drop('season', axis=1, inplace=True); X_test_df.drop('season', axis=1, inplace=True)

TIME_end = time.time()
metrics['splitting_data'] = TIME_end - TIME_start
print("Splitting data done")


## Imputation
TIME_start = time.time()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

X_train_df = X_train_df[np.sort(X_train_df.columns)]; X_test_df = X_test_df[np.sort(X_test_df.columns)]

X_train_df = pd.DataFrame(imputer.fit_transform(X_train_df), columns=X_train_df.columns)
X_test_df = pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns)

TIME_end = time.time()
metrics['imputation'] = TIME_end - TIME_start
print("Imputation done")


## Normalization
TIME_start = time.time()
scaler = StandardScaler()

X_train_df = pd.DataFrame(scaler.fit_transform(X_train_df), columns = X_train_df.columns)
X_test_df = pd.DataFrame(scaler.transform(X_test_df), columns = X_test_df.columns)

TIME_end = time.time()
metrics['normalization'] = TIME_end - TIME_start
print("Normalization done")


## Encoding landUse
landUse = process_coords_landUse( get_tif_info(landUse_datapath, 'LandUse'), landUse_datapath )
landUse_cols = landUse['LandUse'].unique()

TIME_start = time.time()
landuse_data_train = process_landUse(landUse, X_train_df[['x_p','y_p']])
landuse_data_test = process_landUse(landUse, X_test_df[['x_p','y_p']])

X_train_df = X_train_df.join(landuse_data_train.drop(['x_p','y_p'], axis=1))
X_test_df = X_test_df.join(landuse_data_test.drop(['x_p','y_p'], axis=1))

X_train_df = correct_OHE(X_train_df, ['LandUse_'+str(x) for x in landUse_cols])
X_test_df = correct_OHE(X_test_df, ['LandUse_'+str(x) for x in landUse_cols])

TIME_end = time.time()
metrics['encoding_landuse'] = TIME_end - TIME_start


## Encoding Seasons
TIME_start = time.time()

X_train_df = X_train_df.join(pd.get_dummies(train_seasons)); X_test_df = X_test_df.join(pd.get_dummies(test_seasons))

X_train_df = correct_OHE(X_train_df, seasons_encoding.keys()); X_test_df = correct_OHE(X_test_df, seasons_encoding.keys())

X_train_df = X_train_df[np.sort(X_train_df.columns)]; X_test_df = X_test_df[np.sort(X_test_df.columns)]

TIME_end = time.time()
metrics['encoding_seasons'] = TIME_end - TIME_start
  
print("Encoding done")


## ML MODEL
TIME_start = time.time()

num_round = 500
evallist = [(X_train_df, y_train), (X_test_df, y_test)]

xgb_model = xgb.XGBRegressor()
xgb_model = xgb_model.fit(X_train_df, y_train, eval_set=evallist, verbose = 50)


## Feature selection
print("--Columns", X_train_df.columns.values)
thresholds = np.sort(xgb_model.feature_importances_)
print("--Thresholds", thresholds)
print("-------------------")

i_thresh = 0
for thresh in thresholds:
    if (i_thresh % 10 != 0) and (i_thresh < 120):
        i_thresh += 1
        continue
    i_thresh += 1

    selection = SelectFromModel(xgb_model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train_df)
    select_X_test = selection.transform(X_test_df)
    
    evallist = [(select_X_train, y_train), (select_X_test, y_test)]
    
    selection_model = xgb.XGBRegressor(n_estimators=num_round)
    selection_model = selection_model.fit(select_X_train, y_train, eval_set=evallist, verbose = 50)

    y_pred = selection_model.predict(select_X_test)
    y_pred_train = selection_model.predict(select_X_train)
    y_pred = [x if x<100 else 100 for x in y_pred]
    y_pred_train = [x if x<100 else 100 for x in y_pred_train]
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred)
    selected_feat= X_train_df.columns[(selection.get_support())]  
    print("Selected features:", X_train_df.columns[(selection.get_support())])
    print("Threshold=%.3f, n=%d, R2 Score (train)=%.2f, R2 Score (test)=%.2f" % (thresh, select_X_train.shape[1], r2_train, r2_test))


y_pred = xgb_model.predict(X_test_df); y_pred_train = xgb_model.predict(X_train_df)
y_pred_train = [x if x<100 else 100 for x in y_pred_train]; y_pred = [x if x<100 else 100 for x in y_pred]


TIME_end = time.time()
metrics['ML_model'] = TIME_end - TIME_start



R2_train = r2_score(y_train, y_pred_train); R2_test = r2_score(y_test, y_pred)
mae_train = mean_absolute_error(y_train, y_pred_train); mae_test = mean_absolute_error(y_test, y_pred)

metrics['R2_score_on_train'] = R2_train
metrics['R2_score_on_test'] = R2_test
metrics['MAE_score_on_train'] = mae_train
metrics['MAE_score_on_test'] = mae_test

print("XGBoost Metrics:")
print("=="*50)
print('R2 score (on train):', R2_train)
print('R2 score (on test):', R2_test)
print("=="*50)
print('MAE score (on train):', mae_train)
print('MAE score (on test):', mae_test)
print("=="*50)



