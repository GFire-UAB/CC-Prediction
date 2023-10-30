#!/usr/bin/env python
# coding: utf-8
'''
Main script for predicting the Canopy Cover of a set of blocks. Given a directory containing a set of blocks in .laz by argument on the command line, this 
script predicts their Canopy Cover and returns it in a csv containing a set of "x_p", "y_p", "CC", which are the coordenates and the Canopy Cover
prediction.

Made by Alejandro Donaire, Eric Sanchez, Pau Ventura, Francesco Tedesco.
GFIRE.
'''

## IMPORTS
# Data Handling
import pandas as pd
import numpy as np

# Processing data
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Models
import xgboost as xgb

# Own utils
from ML_utils.reading_laz import *
from ML_utils.block_preprocessing_utils import *
from ML_utils.block_preprocessing_CC import *
from ML_utils.block_preprocessing_merge import *
from ML_utils.LandUse_preprocessing import *


# Others / utils
import time
import joblib
from multiprocessing import Process, Queue, cpu_count
from os import listdir
from os.path import isfile, join

import logging
logging.basicConfig(level=logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

import argparse
import rasterio
from rasterio.transform import from_origin


def Main_CC_Prediction(input_blocks_path, output_blocks_path, output_type, user_cpu):
    np.random.seed(0)
    time_start = time.time()

    # Path where the input blocks are stored
    blocks_path = input_blocks_path


    # Constants
    groundtruth_precision = 20          
    outliers = 135

    tif_path = '../data/tifs'
    slope_datapath = join(tif_path,'slope.tif')

    results_path = join('../results', str(8424))
    imputer_path = join(results_path,'model', 'imputer.joblib')
    scaler_path = join(results_path, 'model', 'scaler.joblib')
    xgb_path = join(results_path, 'model','xgb.joblib')
    output_path = join(results_path, 'predictions')

    all_blockfiles = [join(blocks_path, f) for f in listdir(blocks_path) if isfile(join(blocks_path, f)) and f[-4:] == '.laz']
    blockfiles = np.array(all_blockfiles)

    if user_cpu == -1 or user_cpu > cpu_count(): 
        available_cpu = cpu_count()
    else:
        available_cpu = user_cpu
        
    total_processed_blocks = len(blockfiles)
    if total_processed_blocks < available_cpu:
        n_cpu = total_processed_blocks
    else:
        n_cpu = available_cpu


    step = total_processed_blocks//n_cpu

    sampling = np.random.choice(len(blockfiles), len(blockfiles), replace=False) # Ensures replication
    block_batches = sampling[:n_cpu*step].reshape(n_cpu, step)
    other_blocks = sampling[n_cpu*step:]


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

    # Processing other blocks using one cpu per block
    processes = [Process(target=preprocess_blocks_parallel_CC, args=(blockfiles[single_block], groundtruth_precision, queue, outliers))
                for single_block in other_blocks]

    for p in processes:
        p.start()

    for p in processes:
        df_list.extend(queue.get())

    for p in processes:
        p.join()



    ## Reading tiff info
    slope = process_coords_aspect_slope( get_tif_info(slope_datapath, 'slope'), slope_datapath )

    slope['x_p'] = slope['x_p'].astype(np.int32)
    slope['x_p'] = slope['x_p'].astype(np.int32)

    # Merging in parallel
    queue = Queue()

    block_batches_proc = []

    if step != 0: 
        auxlist = df_list[:n_cpu*step]
        block_batches_proc = [auxlist[i:i+step] for i in range(0, len(auxlist), step)]

    processes = [Process(target=parallel_merging_tiffs, args=(batch, slope, queue))
                for batch in block_batches_proc] 

    for p in processes:
        p.start()

    processed_blocks = [] 
    for p in processes:
        processed_blocks.extend(queue.get())

    for p in processes:
        p.join()

    parallel_merging_tiffs(df_list[step*n_cpu:], slope, queue)
    processed_blocks.extend(queue.get())

    del slope, df_list

    ## Correcting NaN
    for i in range(len(processed_blocks)):
        processed_blocks[i][['slope']] = processed_blocks[i][['slope']].replace(-9999, np.nan) # Will be imputed


    ## Final adjustments
    X_test_list = processed_blocks
    coords = [X_test_df[['x_p', 'y_p']] for X_test_df in X_test_list]

    # Imputation
    imputer = joblib.load(imputer_path)

    X_test_list = [X_test_df[np.sort(X_test_df.columns)] for X_test_df in X_test_list]
    X_test_list = [pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns) for X_test_df in X_test_list]

    # Normalization
    scaler = joblib.load(scaler_path)

    X_test_list = [pd.DataFrame(scaler.transform(X_test_df), columns = X_test_df.columns) for X_test_df in X_test_list]
    X_test_list = [X_test_df.drop(['x_p', 'y_p'], axis=1) for X_test_df in X_test_list]


    ## ML MODEL
    xgb_model = joblib.load(xgb_path)

    TIME_start = time.time()
    y_pred_list = [xgb_model.predict(X_test_df) for X_test_df in X_test_list]
    y_pred_list = [[x if x<100 else 100 for x in y_pred] for y_pred in y_pred_list]
    y_pred_list = [[x if x>0 else 0 for x in y_pred] for y_pred in y_pred_list]


    TIME_end = time.time()
    print("Model time:",  TIME_end - TIME_start)

    ## Saving results
    i=1
    for coord, y_pred in zip(coords, y_pred_list):
        coord['Pred_CC'] = y_pred

        x_grid, y_grid = np.meshgrid(np.arange(coord['x_p'].min(), coord['x_p'].max()+groundtruth_precision, groundtruth_precision),
                                np.arange(coord['y_p'].min(), coord['y_p'].max()+groundtruth_precision, groundtruth_precision))

        grid_df = pd.DataFrame({'x_p': x_grid.flatten(), 'y_p': y_grid.flatten()})

        merged_df = pd.merge(grid_df, coord, on=['x_p', 'y_p'], how='left')
        merged_df['Pred_CC'].fillna(-9999, inplace=True) # Filling 20x20m area where we have no pixel values for -9999 CC

        merged_df['Pred_CC'] = merged_df['Pred_CC'].astype(np.float16)
        merged_df = merged_df.sort_values(['y_p','x_p'], ascending =[False, True]) # Necessary order for sending it to tiff
        if output_type == 'csv':
            merged_df.to_csv(join(output_blocks_path, "CC_Prediction_block_"+str(i)+".csv"), index=False)
        elif output_type == 'tiff':
            # define parameters for .tiff file
            width = merged_df['x_p'].nunique() # Number of columns
            height = merged_df['y_p'].nunique()  # Number of rows
            dtype = 'float32'  # can be modified 
            output_pred_path = join(output_blocks_path, "CC_Prediction_block_"+str(i)+".tiff")
            
            # Unique values for 'x' y 'y'
            x_values = merged_df['x_p'].unique()
            y_values = merged_df['y_p'].unique()

            # Get the pixel size
            x_pixel_size = (x_values.max() - x_values.min()) / width
            y_pixel_size = (y_values.max() - y_values.min()) / height

            # Create a tranform object to define the relationship between world coodinates and pixels
            transform = from_origin(x_values.min(), y_values.max(), x_pixel_size, y_pixel_size)
            
            # Create the tiff file
            with rasterio.open(output_pred_path, 'w', driver='GTiff', height=height, width=width,
                    count=1, dtype=dtype, crs='EPSG:4326', transform=transform) as dst:
                    dst.write(merged_df['Pred_CC'].values.reshape((height, width)), 1)
        elif output_type == 'asc': 
            # define parameters for .asc file
            width = merged_df['x_p'].nunique() # Number of columns
            height = merged_df['y_p'].nunique()  # Number of rows
            dtype = 'float32'  # can be modified 
            
            # Unique values for 'x' y 'y'
            x_values = merged_df['x_p'].unique()
            y_values = merged_df['y_p'].unique()

            # Get the pixel size
            x_pixel_size = (x_values.max() - x_values.min()) / width
            y_pixel_size = (y_values.max() - y_values.min()) / height


            output_asc = join(output_blocks_path, "CC_Prediction_block_" + str(i) + ".asc")

            # Reshape 'Pred_CC' values to match the shape of the grid
            pred_cc_values = merged_df['Pred_CC'].values.reshape((height, width))

            # Write to ASCII grid
            with open(output_asc, 'w') as asc_file:
                # Write header
                asc_file.write(f"ncols {width}\n")  
                asc_file.write(f"nrows {height}\n") 
                asc_file.write(f"xllcorner {x_values.min()}\n")  # ll : x-coordinate of lower-left corner
                asc_file.write(f"yllcorner {y_values.min()}\n")  # ll: y-coordinate of lower-left corner
                asc_file.write(f"cellsize {x_pixel_size}\n")  # Cell size in the x-direction (same as y)
                asc_file.write(f"nodata_value 0 \n")  # Value to represent nodata

                # Write data
                for row in pred_cc_values:
                    # Convert the row data to a space-separated string and write it to the file
                    asc_file.write(" ".join(map(str, row)) + "\n")


        else: 
            print("Invalid output type. Supported types are 'csv', 'tiff', and 'asc'.")

        i += 1


    print("Total processing time:", (time.time()-time_start)/60/60, "hours")



def main():
    # Path where the input blocks are stored

    parser = argparse.ArgumentParser(prog='Main_CC_Prediction',
                                     description=
                                     """
                                     Main script for predicting the Canopy Cover of a set of blocks. Given a directory containing 
                                     a set of blocks in .laz by argument on the command line. 
                                        note: By default this script predicts their Canopy Cover and returns it in a csv containing a set of 
                                        "x_p", "y_p", "CC", which are the coordenates and the Canopy Cover prediction.
                                     """, 
                                     epilog='Made by Alejandro Donaire, Eric Sanchez, Pau Ventura, Francesco Tedesco. \nGFIRE.')
    
    parser.add_argument('input_blocks_path', type=str, help='Path where the input blocks are stored')

    parser.add_argument('output_blocks_path', type=str, help='Path where the output blocks will be stored')

    parser.add_argument('--output_type','-o', type=str, choices=['tiff', 'asc', 'csv'], default='csv',
                    help='Output CC block prediction type (default: csv)')
    

    parser.add_argument('--n_cpu','-n', type=int,  default=-1,
                    help='Number of CPU that will be used (default: -1 -> max cpu available)')

    
    args = parser.parse_args()

    input_blocks_path = args.input_blocks_path 
    output_blocks_path = args.output_blocks_path
    output_type = args.output_type
    n_cpu = args.n_cpu


    Main_CC_Prediction(input_blocks_path, output_blocks_path, output_type, n_cpu)


if __name__ == '__main__':
     main()
