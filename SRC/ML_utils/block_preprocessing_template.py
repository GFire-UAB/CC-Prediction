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

logging.basicConfig(level=logging.WARNING)


def preprocess_blocks_parallel(block_list : List[str], gt_precission : int, queue : Queue , outliers : int, seasons_encoding : dict) -> None:
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
    
    season_encoding: 
        - Dictionary encoding seasons where the keys indicate the season. 
    
    
    Returns    
    -------  
    None
    '''

    i=0
    df_list = []
    for my_block in block_list:
        logging.warning("Processing block START: "+ str(multiprocessing.current_process().name)+"_"+str(i))
        
        my_block = read_block(my_block, seasons_encoding) # Reads block        
        feature_list = preprocess_block(my_block, gt_precission, outliers) # Extracts the features from the block
        del my_block

        df_list.append(merge_features(feature_list))
        del feature_list

        i += 1           
    
    queue.put(df_list)





def read_block(block_path : str, seasons_encoding : dict) -> pd.DataFrame:  
    '''
    Reads the block data and adds the season feature.
    
    Parameters
    ----------
    block_path: str 
        - Path names to access the blocks data.

        
    season_encoding: 
        - Dictionary encoding seasons where the keys indicate the season. 
    
    Returns    
    -------  
    pd.DataFrane
        - Dataframe with the block data
    '''

    my_block = get_input_dataframe(block_path) # Reading block
    
    # Adding Season feature
    my_block['season'] = get_seasons(my_block['t'], seasons_encoding)
    my_block.drop('t', inplace=True, axis=1)

    return my_block    


def get_seasons(t_array : np.array, seasons_encoding : dict) -> np.array:
    '''
    Classifies lidar points into seasons depending on when the points were taken

    
    Parameters
    ----------
    t_array: np.array 
        - Numpy array with the time (t) column.
     
    season_encoding: 
        - Dictionary encoding seasons where the keys indicate the season. 
    
    Returns    
    -------  
    np.array
        - Numpy array with LiDAR points classified. 

    '''
    # Changes units from seconds to days
    t_array = (t_array / 60 / 60 / 24) % 365
    t_array = t_array.apply(round)

    # Season distribution
    seasons = dict()
    seasons['spring'] = [80, 172]  # Spring = 20 March to 21 June 
    seasons['summer'] = [172, 264]  # Summer = 21 June to 23 September 
    seasons['autumn'] = [264, 355] # Autum = 23 September to 21 December

    # Point Classification
    seasons_arr = np.zeros(t_array.shape[0])
    for key, value in seasons.items():
        seasons_arr += (t_array > value[0])+0 * (t_array < value[1])+0 * seasons_encoding[key] # Fast vectorized implementation
    
    # Replacing encodings with season names
    seasons_arr = seasons_arr.replace({value:key for key,value in seasons_encoding.items()})

    return seasons_arr


def preprocess_block(my_block : pd.DataFrame, gt_precission : int, outliers : int) -> List[pd.Series]:
    '''
    Does all the operations to preprocess the block. First collects all the pixels into the same point,
    then adds the features. Features create a grid. Requires feature selection.

    Final features: 152

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
    LIDAR_POINT_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 18] # Default classes for a LiDAR

    # Removing outliers
    my_block = my_block.query('c != @outliers')

    # Grouping blocks
    my_block = collect_pixels_LeftTop(my_block, gt_precission)
    
    # Creating the feature extractor object
    FE = FeatureExtractor(my_block, 6)
    
    # Adding the features to the dataset
    not_features = ['x','y','x_p','y_p','height_merging_x', 'height_merging_y', 'c', 'z', 'season']
    classes = LIDAR_POINT_CLASSES
    columns = my_block.columns[ ~my_block.columns.isin(not_features) ].to_list()
    columns.append('height')
    
    features = []
    # Adding quantile and estadistic features
    for column in columns:
        estadistic_features = [ FE.mean_feature(column), FE.sd_feature(column), FE.max_diff_feature(column) ]
        quantile_features = [ FE.quantile_feature(column, q) for q in np.arange(0, 1, 0.1) ]
    
        features.extend(estadistic_features + quantile_features)

    # Adding Threshold features
    for uov in [True, False]:
        threshold_height = [ FE.threshold_percentage_feature('height', threshold, use_only_vegetation = uov) for threshold in np.arange(2, 15, 0.5) ]
        threshold_return = [ FE.threshold_percentage_feature('r', threshold, use_only_vegetation = uov) for threshold in np.arange(1, 5, 1) ]
        threshold_intensity = [ FE.threshold_percentage_feature('i', threshold, use_only_vegetation = uov) for threshold in np.arange(10, 250, 30) ]
    
        features.extend(threshold_height + threshold_return + threshold_intensity)

    # Adding class percentage features
    for point_class in classes:
        features.append( FE.pts_num_feature(point_class) )
    
    features.append(FE.pts_not_classified()) # Not classified points feature

    features.append(FE.seasons_feature()) # Adding Points seasons

    for radius in np.arange(2, 10, 0.5):
        features.append(FE.num_tree_tops(radius)) # Number of tree tops estimate

    del FE

    return features
