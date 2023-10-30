'''
Script for predicting the whole Catalunya using a given model.

Alejandro Donaire, Eric Sanchez, Pau Ventura, Francesco Tedesco.
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
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper

# Own utils
from ML_utils.reading_laz import *
from ML_utils.block_preprocessing_utils import *
from ML_utils.block_preprocessing_CC import *
from ML_utils.block_preprocessing_merge import *
from ML_utils.LandUse_preprocessing import *

# Others / utils
import time
import csv
import joblib
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from ML_utils.block_preprocessing import *
from ML_utils.reading_laz import *
from multiprocessing import Process, Queue
from os import listdir
from os.path import isfile, join
import psutil
import pprint

import logging
logging.basicConfig(level=logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)

time_start = time.time()

# Constants
n_cpu = 19
groundtruth_precision = 20          
outliers = 135
total_processed_blocks = n_cpu*444 # Must be multiple of n_cpu, currently 8424 out of 8436
step = total_processed_blocks//n_cpu

data_path = '/lex/shared/projects/nnveget/data/laz'
control_test_path = '/lex/shared/projects/nnveget/data/control_test_laz'
tif_path = '/lex/shared/projects/nnveget/data/tifs'
groundtruth_datapath = join(tif_path,'variables-biofisiques-arbrat-v1r0-cc-2016-2017.tif')
slope_datapath = join(tif_path,'slope.tif')

results_path = join('/lex/shared/projects/nnveget/results', str(8424))
imputer_path = join(results_path,'model', 'imputer.joblib')
scaler_path = join(results_path, 'model', 'scaler.joblib')
xgb_path = join(results_path, 'model','xgb.joblib')
output_path = join(results_path, 'predictions')


## Stochastic block sampling
all_blockfiles = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
control_test_blockfiles = [join(control_test_path, f) for f in listdir(control_test_path) if isfile(join(control_test_path, f))]
blockfiles = np.array(all_blockfiles)

assert total_processed_blocks % n_cpu == 0, "ERROR: Processed blocks must be a multiple of ("+str(n_cpu)+")"
print("Processing:", total_processed_blocks, "blocks")

sampling = np.random.choice(len(blockfiles), len(blockfiles), replace=False) # Ensures replication
block_batches = sampling[:total_processed_blocks].reshape(n_cpu, step)

print("Block sampling done")
print("Blocks used:", block_batches)
print("Ram usage (after block sampling):", psutil.virtual_memory()[2])


## PARALLEL block preprocessing
queue = Queue()
  
processes = [Process(target=preprocess_blocks_parallel_CC, args=(blockfiles[batch], groundtruth_precision, queue, outliers))
             for batch in block_batches]

for p in processes:
    p.start()

df_list = []
for p in processes:
    df_list.extend(queue.get())

for p in processes:
    p.join()

print("Ram usage (block preprocessing done):", psutil.virtual_memory()[2])


## Reading other tiff info
groundtruth = get_tif_info(groundtruth_datapath, 'CC')
groundtruth['x_p'] = groundtruth['x_p'].astype(np.int32)
groundtruth['y_p'] = groundtruth['y_p'].astype(np.int32)
gt_filter = groundtruth['CC'] > 100
groundtruth['CC'] = gt_filter * 100 + (~gt_filter) * groundtruth['CC'] # Fast vectorized implementation
	
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

df_list = processed_blocks
del processed_blocks

slope = process_coords_aspect_slope( get_tif_info(slope_datapath, 'slope'), slope_datapath )

slope['x_p'] = slope['x_p'].astype(np.int32)
slope['x_p'] = slope['x_p'].astype(np.int32)


## Merging in parallel
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

del slope, df_list
print("Ram usage (after merging slope):", psutil.virtual_memory()[2])

# Correcting NaN
for i in range(len(processed_blocks)):
    processed_blocks[i][['slope']] = processed_blocks[i][['slope']].replace(-9999, np.nan) # Will be imputed
    processed_blocks[i][['CC']] = processed_blocks[i][['CC']].fillna(0)


## Splitting data
X_test_list = processed_blocks

X_test_df = pd.concat(X_test_list, axis=0).reset_index(drop=True)

y_test = X_test_df['CC']
coords = X_test_df[['x_p', 'y_p']]

X_test_df = X_test_df.drop('CC', axis=1)
print("Splitting data done")


# Imputation
imputer = joblib.load(imputer_path)

X_test_df = X_test_df[np.sort(X_test_df.columns)]

X_test_df = pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns)
print("Imputation done")


# Normalization
scaler = joblib.load(scaler_path)

X_test_df = pd.DataFrame(scaler.transform(X_test_df), columns = X_test_df.columns)

print("Normalization done")

X_test_df = X_test_df.drop(['x_p', 'y_p'], axis=1)


# ML MODEL
xgb_model = joblib.load(xgb_path)

TIME_start = time.time()
y_pred = xgb_model.predict(X_test_df)
y_pred = [x if x<100 else 100 for x in y_pred]
y_pred = [x if x>0 else 0 for x in y_pred]

TIME_end = time.time()
print("Model time:",  TIME_end - TIME_start)

R2_test = r2_score(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)

print("XGBoost Metrics:")
print("=="*50)
print('R2 score (on test):', R2_test)
print("=="*50)
print('MAE score (on test):', mae_test)
print("=="*50)

coords['Real CC'] = y_test
coords['Pred CC'] = y_pred
coords['Pred CC'] = coords['Pred CC'].astype(np.float16)
coords.to_csv(join(output_path, "Full_Cat.csv"), index=False)


print("Total processing time:", (time.time()-time_start)/60/60, "hours")
