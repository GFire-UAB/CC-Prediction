import numpy as np
import pandas as pd
from multiprocessing import Queue
import psutil
import logging
from ML_utils.block_preprocessing_utils import *

logging.basicConfig(level=logging.WARNING)


def parallel_merging_tiffs(batch : pd.DataFrame, tiff_df : pd.DataFrame, queue : Queue) -> None:
    '''
    Wrapper function that merges the block data with tiff data using CPU paralelization.
    
    Parameters
    ----------
    batch : pd.DataFrame 
        - Pandas dataframe with several blocks data.

    tiff : pd.DataFrame
        - Pandas dataframe with the tiff data

    queue : Queue
        - Queue object to store pd.DataFrames during the parallelization process.
    
    Returns    
    -------  
    None
    '''
    processed_batch = []
    for i in np.arange(len(batch)):
        batch[i] = merge_tiff(batch[i], tiff_df) # Adding tiff information to each block
        processed_batch.append(batch[i])
       
        logging.warning("Ram usage after adding tiff info:"+str(psutil.virtual_memory()[2]))
        
    queue.put(processed_batch)


def merge_tiff(features_df : pd.DataFrame, tiff_df : pd.DataFrame) -> pd.DataFrame: 
    """
    Function that merges the block data with tiff data.
    
    Parameters
    ----------
    features_df : pd.DataFrame 
        - Pandas dataframe with the block data.

    tiff_df : pd.DataFrame
        - Pandas dataframe with the tiff data. 

    Returns    
    -------  
    pd.DataFrame: 
        - Resulting dataframe once merged. 
    """
    features_coords = features_df[['x_p','y_p']].rename({'x_p':'x','y_p':'y'}, axis=1) # Getting the corrds of the block
    
    chunk_size = 5_000_000 # size of chunks. Processing by chunks saves up RAM
    chunks = [tiff_df[i:i + chunk_size] for i in range(0, tiff_df.shape[0], chunk_size)] # Splitting the tiff into chunks
    
    tiff_filtered = pd.concat([filter_df(chunk, features_coords) for chunk in chunks]) # Filter the tiff according to the block coords
    features_df = features_df.merge(tiff_filtered, how='left', on=['x_p','y_p']) # Adds the tiff feature to the block
   
    return features_df    




