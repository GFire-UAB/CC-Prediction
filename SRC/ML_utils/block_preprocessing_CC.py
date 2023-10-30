import pandas as pd
import numpy as np
from multiprocessing import Queue
from typing import List
import logging
import multiprocessing
import psutil

from ML_utils.FE_class import *
from ML_utils.reading_laz import *
from ML_utils.block_preprocessing_utils import *


def preprocess_blocks_parallel_CC(block_list : List[str], gt_precission : int, queue : Queue , outliers : int) -> None:
    '''
    Wrapper function that preprocesses the block using CPU paralelization.
    
    Parameters
    ----------
    block_list : List[str] 
        - List with the path names to access the blocks data.
    gt_precision : int
        - Indicates the block height/width precision.
    queue : Queue
        - Queue object to store pd.DataFrames during the parallelization process.
    outliers : int
        - Outlier LiDAR class
    
    Returns    
    -------  
    None
    '''
    i=0
    df_list = []
    for my_block in block_list:
        my_block = read_block_CC(my_block)

        feature_list = preprocess_block_CC(my_block, gt_precission, outliers)
        del my_block

        df_list.append(merge_features(feature_list))
        del feature_list

        i += 1

    queue.put(df_list)




def read_block_CC(block_path : str) -> pd.DataFrame:
    '''
    Reads the block data and drops the t feature
    
    Parameters
    ----------
    block_path: str 
        - Path names to access the blocks data.

    Returns    
    -------  
    pd.DataFrane
        - Dataframe with the block data
    '''
    my_block = get_input_dataframe(block_path) # Reading the block
    my_block.drop('t', inplace=True, axis=1) # Dropping the t column as won't be used for CC prediction
    
    return my_block    


def preprocess_block_CC(my_block : pd.DataFrame, gt_precission : int, outliers : int) -> List[pd.Series]:  
    '''
    Does all the operations to preprocess the block. First collects all the pixels into the same point,
    then adds the features. (Final features: 16 )

    Parameters
    ----------
    my_block : pd.DataFrame 
        - Pandas dataframe with the blocks data.
    gt_precision : int
        - Indicates the block height/width precision.
    outliers : int
        - Outlier LiDAR class 
    
    Returns    
    -------  
    List[pd.Series]
        - List of pd.Series for each feature extracted.
    '''
    # Removing outliers
    my_block = my_block.query('c != @outliers')

    # Grouping blocks
    my_block = collect_pixels_LeftTop(my_block, gt_precission)

    # Creating the feature extractor object
    FE = FeatureExtractor(my_block, 6)

    # Adding the features to the dataset
    features = []

    features.extend( [ FE.threshold_percentage_feature('height', threshold, use_only_vegetation = True) for threshold in np.arange(3, 8) ] )
    features.append( FE.threshold_percentage_feature('height', 3, use_only_vegetation = True))   
    
    features.extend([ FE.quantile_feature('height', quant) for quant in [0.2, 0.3] ])
    
    features.extend( [ FE.quantile_feature('n', quant) for quant in np.arange(0.3, 0.8, 0.1)] )

    features.append(FE.threshold_percentage_feature('i', 10, use_only_vegetation=True))
    features.append(FE.threshold_percentage_feature('i', 70, use_only_vegetation=False))
    features.append(FE.threshold_percentage_feature('i', 130, use_only_vegetation=False))

    features.append(FE.sd_feature('n'))
   
    features.append(FE.pts_num_feature(2))
    features.append(FE.pts_num_feature(5))

    features.append(FE.threshold_percentage_feature('r', 1, use_only_vegetation=False))
    
    del FE

    return features
