import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

class FeatureExtractor:
    """
    Extracts the features of each point based on the pixels inside it.
    
    Attributes
    ----------
    input_data : Pandas DataFrame
        LiDAR input information with the coordinates of the points corresponding to each pixel.

    block_height_grouping: int
        Size of the side of the square that is estimated to have constant height. Helps estimate the height of the points.
    
    Methods
    -------
    num_tree_tops(radius=4)
    mean_feature(feature)
    quantile_feature(feature, quant=0.5)
    sd_feature(feature)
    max_diff_feature(feature)
    threshold_percentage_feature(feature, threshold, use_only_vegetation)
    pts_num_feature(number)
    pts_not_classified(name='not_classified_pts')
    """
    
    def __init__(self, input_data: pd.DataFrame, block_height_grouping : int = 6) -> None:
        """
        Initializes a new FeatureExtractor object.

        Parameters:
        ----------
        input_data : pd.DataFrame 
            LiDAR input information with the coordinates of the points corresponding to each pixel.
        
        block_height_grouping : int 
            Size of the side of the square that is estimated to have constant height. 
            Helps estimate the height of the points.
        
        Returns
        -------
        None   
        """
        if not set(['x_p', 'y_p', 'x', 'y', 'c', 'a']).issubset(set(input_data.columns)):
            print("Error: Input data does not have the required columns")
            
        self.xp = 'x_p'      # pixel's point x coordinate column name
        self.yp = 'y_p'      # pixel's point y coordinate column name
        self.x = 'x'         # pixel x coordinate column name
        self.y = 'y'         # pixel y coordinate column name
        self.c = 'c'         # pixel class column name
        self.a = 'a'         # pixel angle columns name
        self.r = 'r'         # pixel return number columns name
        self.i = 'i'

        self.data = self._get_heights(input_data, block_height_grouping)  
        del input_data
        self.grouped = self.data.groupby([self.xp, self.yp])  # pixels grouped by point
        self.points = self.grouped.count()[[self.y]].reset_index()              
    
    def _get_heights(self, input_data: pd.DataFrame, block_height_grouping: int) -> pd.DataFrame:
        """
        Estimates the height of each point

        Parameters:
        ----------
        input_data : pd.DataFrame 
            LiDAR input information with the coordinates of the points corresponding to each pixel.
        
        block_height_grouping : int
            Size of the side of the square that is estimated to have constant height. 
            Helps estimate the height of the points.
        
        Returns
        -------
        pd.DataFrame
            LiDAR data with point-assigned pixels and its height
        """
        input_data['height_merging_x'] = ( input_data[self.x] // block_height_grouping ) * block_height_grouping
        input_data['height_merging_y'] = ( input_data[self.y] // block_height_grouping ) * block_height_grouping

        input_copy = input_data.copy()

        input_copy = input_copy.groupby(['height_merging_x', 'height_merging_y'])['z'].min().reset_index()
        input_copy = input_copy.rename({'z':'surface_z'}, axis=1)

        input_data = pd.merge(input_data, input_copy, how='left', on=['height_merging_x', 'height_merging_y'])
        input_data['height'] = input_data['z'] - input_data['surface_z']

        input_data = input_data.drop(['z', 'surface_z', 'height_merging_x', 'height_merging_y'], axis=1)
        
        return input_data
    
    def _get_tree_tops(self, vect: pd.DataFrame, radius: int = 4) -> int:
        '''
        Number of tree tops

        Parameters
        ----------
        vect : pd.DataFrame 
            dataframe 
        radius : int
            small radius to check if it is a local maximum

        Returns
        -------
        int
            Resulting number of tree tops.
        '''
        vect = vect[['x','y','height']]
        vect = vect.values
        
        sorted_vect = np.argsort(vect[:,2])[::-1]
        tops = np.array([sorted_vect[0]])
        dist = cdist(vect[:,0:2], vect[:,0:2])
        
        for i in sorted_vect[1:]:
            near = np.where(dist[i] < radius)[0]
            for n in near:
                if vect[n][2] > vect[i][2]:
                    break
            else:
                tops = np.append(tops, i)
        return len(tops)

    
    def num_tree_tops(self, radius: int = 4, name: str = 'num_tops') -> pd.Series:
        '''
        Number of tree tops / local maximums (estimated according to custom algorithm)

        Parameters
        ----------
        radius : int
            small radius to check if it is a local maximum
        name : str 
            name of the resulting series object (that will eventually be the column name in the dataset)

        Returns
        -------
        pd.Series
            Resulting series with the column of representing the number of tree tops / local maximums.
        '''
        filtered = self.data.query('c in [4,5]').copy()
        values = filtered.groupby([self.xp,self.yp]).apply(self._get_tree_tops, radius=radius)
        values = pd.DataFrame(values, columns=[name+'_'+str(radius)])
        return values     

    # Return oriented features
    def mean_feature(self, feature: str) -> pd.Series:
        '''
        Calculates the mean of the given feature for each grouped pixels.
        
        Parameters
        ----------
        feature : str 
            The column name to get the feature.
        
        Returns    
        -------
        pd.Series
            Resulting Series with the means of the corresponding grouped pixel.
        '''
        values = self.grouped[feature].mean()
        return values.rename(str(feature)+'_mean')
    

    def quantile_feature(self, feature: str, quant: float=0.5) -> pd.Series:
        """
        Calculates the given quantile of the given feature for each grouped pixels.
        
        Parameters
        ----------
        feature : str
            The column name to get the feature.
        quant : float
            Percentage of the quantile. Range: [0,1]
        Returns    
        -------
        pd.Series
            Resulting Series with the quantiles of the corresponding grouped pixel.        
        """
        values = self.grouped[feature].quantile(quant)
        return values.rename(str(feature)+'_q_'+str(round(quant,2)))
    

    def sd_feature(self, feature: str) -> pd.Series:
        '''
        Calculates the standard deviation of the given feature for each grouped pixels.
        
        Parameters
        ----------
        feature : str 
            The column name to get the feature.

        Returns    
        -------                
        pd.Series
            Resulting Series with standard deviations of the corresponding grouped pixel. 
        '''
        values = self.grouped[feature].std()
        return values.rename(str(feature)+'_sd')
    

    def max_diff_feature(self, feature: str) -> pd.Series:
        """
        Calculates the difference between the highest and the lowest value of the given feature for each grouped pixels.
        
        Parameters
        ----------
        feature : str
            The column name to get the feature.
        
        Returns    
        -------                
        pd.Series
            Resulting Series with the difference between the highest and the lowest value of for each grouped pixels.
        """
        values = self.grouped[feature].max() - self.grouped[feature].min()
 
        return values.rename(str(feature)+'_max_diff')
    

    def threshold_percentage_feature(self, feature: str, threshold: float, use_only_vegetation: bool) -> pd.Series:
        """
        Applies a filter to only keep those points where feature > threshold. Calculates the number of points that have 
        a feature value above a certain threshold.
        
        Parameters
        ----------
        feature : str
            The column name to get the feature.
        threshold : float
            Threshold value to filter the points. 
        use_only_vegetation : bool
            To apply the filter to the points that are only classified as vegetation. 
        
        Returns    
        -------   
        pd.Series
            Resulting series with the corresponding counts. 
        """
        my_block = self.data.copy()
        df_canopy = my_block[ my_block[feature] > threshold ].copy()
        if use_only_vegetation:
            df_canopy = df_canopy.query('c in [3,4,5]')

        df_canopy['counted_canopy'] = np.zeros(df_canopy.shape[0])
        df_canopy = df_canopy.groupby([self.xp, self.yp])[['counted_canopy']].count().reset_index()

        my_block['counted_no_canopy'] = np.zeros(my_block.shape[0])
        my_block = my_block.groupby([self.xp, self.yp])[['counted_no_canopy']].count().reset_index()

        my_block = pd.merge(my_block, df_canopy, how='left', on = [self.xp, self.yp])

        my_block['counted_canopy'].fillna(0, inplace=True)
        my_block[str(feature)+'_threshold_'+str(threshold)+'_'+str(use_only_vegetation)] = 100*my_block['counted_canopy'] / my_block['counted_no_canopy']
        
        my_block.drop(['counted_canopy', "counted_no_canopy"], axis=1, inplace = True)
        
        return my_block
   

    def pts_num_feature(self, number: int) -> pd.Series:
        '''
        Calculates the percentage of pixels on the point that belong to a given class number.

        Parameters
        ----------
        number : int 
            Number of the class to extract the feature.

        Returns    
        -------   
        pd.Series
            Resulting series with the corresponding percentages.  
        '''
        filtered = self.data.query('c == @number').copy()
        filtered = filtered.groupby([self.xp,self.yp]).count()[[self.x]].reset_index()

        values = self.points.merge(filtered, how='left', on = [self.xp, self.yp])
        values['num_points_'+str(number)] = values[self.x]/values[self.y]

        return values.drop([self.x, self.y], axis=1)
    

    def pts_not_classified(self, name: str='not_classified_pts') -> pd.Series:
        '''
        Calculates the number of points that are not vegetation nor ground.
        
        Parameters
        ----------
        name : str 
            name of the resulting series object (that will eventually be the column name in the dataset)

        Returns    
        -------   
        pd.Series
            Resulting series with the number of points that are not classified as vegetation nor ground. 
        '''
        filtered = self.data.query('c not in [3, 4, 5, 6, 7]').copy()
        filtered = filtered.groupby([self.xp,self.yp]).count()[[self.x]].reset_index()

        values = self.points.merge(filtered, how='left', on = [self.xp, self.yp])
        values[name] = values[self.x]/values[self.y]

        return values.drop([self.x, self.y], axis=1)


    def seasons_feature(self) -> pd.DataFrame:
        '''
        Assigns a season to each point using the most frequent one.
        '''
        return self.grouped['season'].agg(lambda x: pd.Series.mode(x)[0])
