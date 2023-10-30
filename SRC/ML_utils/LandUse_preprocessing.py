import pandas as pd
import numpy as np

from ML_utils.block_preprocessing_utils import *

def process_coords_landUse(landUse : pd.DataFrame, datapath : str) -> pd.DataFrame:  
    """
    Grouping into 20x20. Those even coords get subtracted/added 10 points. 
    
    Parameters
    ----------
    landUse : pd.DataFrame 
        - Pandas dataframe with LandUse data grouped by 10 * 10 m cells
    
    datapath : str
        - String indicating the LandUse datapath
    
    Returns    
    -------  
    pd.DataFrame  
    """
    landUse['LandUse'] = landUse['LandUse'].astype(np.ubyte)

    standarize_coords(landUse, datapath)  

    landUse['x_p'] -= landUse['x_p'] % 20
    landUse['y_p'] += landUse['y_p'] % 20

    landUse['x_p'] = landUse['x_p'].astype(np.int32)
    landUse['y_p'] = landUse['y_p'].astype(np.int32)

    return landUse


def filter_landUse(landUse : pd.DataFrame, my_block: pd.DataFrame) -> pd.DataFrame:
    """
    Changes the data types of the block and landUse to optimize memory usage   
    
    Parameters
    ----------
    landUse : pd.DataFrame 
        - Pandas dataframe with LandUse data grouped by 10 * 10 m cells
    
    my_block : pd.DataFrame
        - Pandas data with LiDAR data
    
    Returns    
    -------  
    pd.DataFrame
        - Pandas dataframe from filter_df function
    """
    landUse['LandUse'] = landUse['LandUse'].astype(np.ubyte)
    landUse['x_p'] = landUse['x_p'].astype(np.int32)
    landUse['y_p'] = landUse['y_p'].astype(np.int32)

    return filter_df(landUse, my_block)


def process_landUse(landUse : pd.DataFrame, aux_df : pd.DataFrame) -> pd.DataFrame:
    """
    Applies the One Hot Encoding process to treat LandUse   
    
    Parameters
    ----------
    landUse : pd.DataFrame 
        - Pandas dataframe with LandUse data grouped by 10 * 10 m cells
    
    aux_df : pd.DataFrame
        - Pandas dataframe 
    
    Returns    
    -------  
    pd.DataFrame
        - Pandas dataframe with the new columns created
    """
    landUse = filter_landUse(landUse, aux_df[['x_p','y_p']].rename({'x_p':'x', 'y_p':'y'}, axis=1))
    landUse = landUse.join(pd.get_dummies(landUse['LandUse'], prefix='LandUse'))

    for col in landUse[landUse.columns[3:]]:
        landUse[col] *= 1/4 # There are 4 10x10 instances inside each 20x20 square (normalization)
        one_column = landUse.groupby(['x_p','y_p'])[[col]].sum().reset_index() # Adding info from different pixels
        aux_df = pd.merge(aux_df, one_column, on=['x_p','y_p'], how='left')

    return aux_df
