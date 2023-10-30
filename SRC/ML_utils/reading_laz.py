import pandas as pd
import numpy as np
import laspy
import rasterio


def get_input_dataframe(laz_file_path):
    """
    Reads a LAS/LAZ file from the given file path and returns a pandas DataFrame containing
    the lidar attributes.

    Args:
        laz_file_path (str): The path of the LAS/LAZ file to read.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - x: x-coordinates of the lidar points.
            - y: y-coordinates of the lidar points.
            - z: z-coordinates of the lidar points.
            - c: classification of the lidar points.
            - a: scan angle of the lidar points.
            - n: number of returns of the lidar points.
            - r: return number of the lidar points.
            - i: intensity of the lidar points.
            - t: time when the lidar points were measured.
    """

    # Read laz file
    file = laspy.read(laz_file_path, laz_backend = laspy.LazBackend.Laszip)

    # Store lidar attributes in a dictionary
    data = {
        "x": np.array(file.x, dtype=np.intc),
        "y": np.array(file.y, dtype=np.intc),
        "z": np.array(file.z, dtype=np.short),
        "c": np.array(file.classification, dtype=np.ubyte),
        "a": np.array(file.scan_angle_rank, dtype=np.byte),
        "n": np.array(file.number_of_returns, dtype=np.ubyte),
        "r": np.array(file.return_number, dtype=np.ubyte),
        "i": np.array(file.intensity, dtype=np.ubyte),
        "t": np.array(file.gps_time)
    }

    # Convert data to pandas DataFrame
    input_df = pd.DataFrame(data)

    return input_df


def get_tif_info(tif_path: str, column_name: str = "z"):
    """
    Reads a .tif file and returns the x, y and z values in three columns as a Pandas
    dataframe. The name of the z value can be specified as a parameter. Uses the
    raterio library.

    Parameters:
    tif_path (str): The path of the complete groundtruth in .tif format.
    column_name (str): Name of the column of the value associated to the x and y
                       coordinates. It is "z" by default.

    Returns:
    pd.DataFrame: Pandas DataFrame with the canopy cover and its coordinates.
    """

    # Open the dataset
    dataset = rasterio.open(tif_path)

    # Read the raster data
    data = dataset.read()

    # Get arrays of x and y pixel coordinates
    j, i = np.mgrid[:dataset.height, :dataset.width]

    # Convert pixel coordinates to geographic coordinates
    x, y = dataset.transform * (i, j)

    # Create a DataFrame with this information
    df_geo = pd.DataFrame({
        'x': x.ravel(),
        'y': y.ravel(),
        column_name: data[0].ravel()
    })
    
    return df_geo.rename({'x':'x_p', 'y':'y_p'}, axis=1)
