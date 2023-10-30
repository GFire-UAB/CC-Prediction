import numpy as np
import pandas as pd
import rasterio
from typing import List


def collect_pixels_LeftTop(my_block : pd.DataFrame, gt_precission : int) -> pd.DataFrame:
    '''
    Given a groundtruth precision, moves all de pixes to the left top of the square for a block.
    Tiff files follow this convention so block pixels have to also be grouped this way.
    Fast vectorized implementation is mandatory.

    Parameters
    ----------
    my_block : pd.DataFrame 
        - Pandas dataframe with the blocks data.
    
    gt_precision : int
        - Indicates the block height/width precision.
    
    Returns    
    -------  
    pd.DataFrame
        - Pandas dataframe with the left top collected pixels.  
    '''
    point_p_d2 = gt_precission/2

    # Adjusting point refference to the center
    aux_xp = my_block['x'] // point_p_d2
    aux_yp = my_block['y'] // point_p_d2

    my_block['x_p'] = aux_xp * point_p_d2 - point_p_d2 * (aux_xp % 2)
    my_block['y_p'] = point_p_d2 + aux_yp * point_p_d2 + point_p_d2 * ((aux_yp % 2==0)+0) 

    # Changing datatype
    my_block['x_p'] = my_block['x_p'].astype(np.int32)
    my_block['y_p'] = my_block['y_p'].astype(np.int32)

    return my_block


def merge_features(features : pd.DataFrame) -> pd.DataFrame:
    '''
    Merges features on x_p and x_y index column.   

    Parameters
    ----------
    features : pd.DataFrame 
        - Pandas dataframe containing the features of each cell.
    
    Returns    
    -------  
    pd.DataFrame
        - Pandas dataframe with grouped features.  
    '''
    grouped_data = features[0].reset_index()[['x_p','y_p']]

    for feature in features[1:]: # features[0] already used above
        grouped_data = grouped_data.merge(feature, how='outer', on=['x_p', 'y_p'])

    grouped_data.fillna(0, inplace=True)

    return grouped_data


def process_coords_aspect_slope(df : pd.DataFrame, datapath : str) -> pd.DataFrame:
    '''
    Wrapper function that processes the data coordinates for aspect and slope.   

    Parameters
    ----------
    df : pd.DataFrame 
        - Pandas dataframe containing the aspect or slope data.

    datapath : str 
        - Path to the data needed.
            
    
    Returns    
    -------  
    pd.DataFrame
        - Pandas dataframe with aspect or slope data processed.  
    '''
    standarize_coords(df, datapath)

    return df


def standarize_coords(df : pd.DataFrame, datapath : str) -> pd.DataFrame:
    '''
    Rounds the x_p and y_p coordinates to the closer point. 
    E.g. if coord_diff == 10, then 4248.54 -> 4250 || 4242.82 -> 4240
    TODO: Add fast vectorized implementation.
    
    Parameters
    ----------
    df : pd.DataFrame 
        - Pandas dataframe containing the aspect or slope data.

    datapath : str 
        - Path to the data needed.
            
    
    Returns    
    -------  
    pd.DataFrame
        - Pandas dataframe with processed coordinates.  
    '''

    check_coords(datapath) # Checks consistency

    with rasterio.open(datapath) as src:
        coord_step = src.meta['transform'][0]

    for coord in ['x_p', 'y_p']:
        to_round = df.iloc[0][coord] % coord_step
        if to_round > coord_step/2: # Coord is closer to the next step
            df[coord] += coord_step - to_round
        else: # Coord is closer to the previous step
            df[coord] -= to_round

    return df



def check_coords(datapath : str) -> None:
    """
    Makes sure that the tiff has a squared-shape and has its origin on the top-left corner.
    
    
    Parameters
    ----------
    datapath : str 
        - Path to the data needed.
            
    
    Returns    
    -------  
    None  
    """
    with rasterio.open(datapath) as src:
        assert src.meta['transform'][0] > 0 and src.meta['transform'][4] < 0 and src.meta['transform'][0] == -src.meta['transform'][4], "ERROR! Tiff metadata is wrong"


def correct_OHE(df : pd.DataFrame, all_cols : List[str]) -> pd.DataFrame:
    """
    Adds columns with 0 to the dataframe to get a coherent One-Hot_Encoding    
    
    Parameters
    ----------
    df : pd.DataFrame 
        - Pandas dataframe to add columns
            
    all_cols : List[str]
        - list of column names to apply the correction
        

    Returns    
    -------  
    pd.DataFrame  
        - Resulting pandas dataframe corrected
    """
    
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.zeros(df.shape[0])

    return df


def filter_df(tiff : pd.DataFrame, my_block : pd.DataFrame) -> pd.DataFrame:
    """
    Cuts the tiff acording to the limits of block coords.
    
    Parameters
    ----------
    tiff : pd.DataFrame 
        - Pandas dataframe with the tiff data. 

    my_block : pd.DataFrame
        - Pandas dataframe with the block data.

    Returns    
    -------  
    pd.DataFrame: 
        - Resulting dataframe with the filtered data. 
    """
    min_x, max_x = my_block['x'].min(), my_block['x'].max()
    min_y, max_y = my_block['y'].min(), my_block['y'].max()

    return tiff.loc[ (tiff['x_p'] < max_x+20) & (tiff['x_p'] > min_x-20) & (tiff['y_p'] < max_y+20) & (tiff['y_p'] > min_y-20) ].copy()   

