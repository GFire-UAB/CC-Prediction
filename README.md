# GFire-CC-prediction

This is part of a bigger project called GFire (https://github.com/GFire-UAB). The main goal of GFire is to predict how a forest fire will spread throughout time in order to help firefighters stop the fire.

This model needs to be fed with some parameters, such as humidity of the zone, canopy cover, fuel types, ... 
Our work consists in building Machine Learning models to predict some of the parameters using LiDAR (high resolution laser) data. 

We have finished working on Canopy Cover and we are able to predict it with a Mean Absolute Error of ~6% on unseen data. Canopy Cover is described as the percentage of the area that is covered with canopy as seen from the sky (ranging from 0% to 100%)

### Datasets information

We are using data from the ICGC (Institut Cartogràfic i Geològic de Catalunya) and the open-source geographic information system QGIS. The data corresponds to different regions of Catalonia (Spanish autonomous community).

### Files information

Folders will be displayed in **bold** and scripts in *italic*.

- **SRC**: Contains all the code files.
    - **ML_utils**: Contain all functions used in the SRC scripts.
        - *block_preprocessing_CC*: Contains functions for preprocessing the blocks when predicting Canopy Cover.
        - *block_preprocessing_merge*: Contains functions for merging .tiff files with the blocks.
        - *block_preprocessing_template*: Contains generic functions for preprocessing blocks to extract a grid of features.
        - *block_preprocessing_utils*: Other utils used while preprocessing the blocks.
        - *FE_class*: Feature Extractor class that allows getting features from a block.
        - *LandUse_preprocessing*: Functions used specifically for preprocessing LandUse tiff.
        - *reading_laz*: Functions used for reading .laz files. Will be used to read the blocks.
    - *Hyperparameter_Search_CC*: Performs hyperparameter search to achieve the best possible model.
    - *FS_Template*: Performs feature selection to filter those features that have a higher impact on predicting the target.
    - *Main_CC_Prediction*: Main script. Given the path of a folder with a set of blocks in .laz format, it calculates the Canopy Cover of each 20mx20m area and returns the output for each block in a separate file on the same folder. Uses all available CPUs.
    - *predicting_blocks_CC*: Uses the CC model to yield a prediction on a set of blocks. Used on Diverse10 dataset.
    - *predicting_cat_CC*: Uses the CC model to yield a prediction in all blocks of Catalonia.
- **data**: Contains all the data used in the scripts.
    - **control_test_laz**: Contains 10 blocks with different geograpical regions. They act as a simplification of Catalonia to check if the model is performing well in different scenarios.
    - **laz**: Contains all the Catalonia blocks. In total 8436. Blocks contain pixels in an area of 2kmx2km.
    - **tifs**: Contains all tifs required to add features to the model such as Slope, Aspect and LandUse. Also contains the groundtruth such as CC.
- **results**: Contains metrics and other important information of script executions. The name of the folder indicates the ammount of blocks used in the script.
    - **8424**: Contains information about the main execution using almost all Catalonia.
        - **metrics**: Contain stats about the execution times of the diferent parts of the script (preprocessing, training, ...)
        - **model**: Contains the trained imputer, scaler and xgb (model) used.
        - **vizz**: Contains the coordinates of the blocks used for traning and test to check their distribution along Catalonia.
- **Updated Slides**: Contains slides of the progress followed week by week with the updates.
- **env**: Contains the packages required to run the code.

### Workflow
For a given biophysical variable (such as Canopy Cover), the workflow through scripts will be the following:
- First, *FS_Template* will be used with a representative subset of the total blocks to detect those features that are relevant for predict the target variable. From now on, only those features will be extracted from the blocks when predicting this target in order to increase accuracy and reduce overfitting, execution time and memory usage.
- Then, *Hyperparameter_Search_<target>* will be created and used to find the best hyperparameters for the model. It is highly advised to use here a large subset of the total available blocks as speed should not be a problem due to the filtering done in the previous step. Then the model will be trained with the best hyperparameters found and using 80% of the available data. The trained model and other necessary instances will be saved to be used 'a posteriori'.
- Finally, *Main_<target>_Prediction* script will be created. Given the path of a folder containing multiple blocks in .laz format, this script will read the blocks, process them, read the previously saved model and use it to predict the target on the readed blocks. The output will be displayed on the same folder as the blocks in the desired format (csv, tif or asc).

Note that only the last script will be available for the client as all other scripts are part of the process to create/train the model.

### Main Usage

In order to get the CC prediction of a certain block, or a set of blocks, you can use the following command structure

python Main_CC_Prediction.py path_to_input_blocks_data path_to_input_blocks_data --output_type your_output_type --n_cpu personalized number of cpu

- __input_blocks_path__: path where the input block / s are stored
- __output_blocks_path__: path where the input block / s will be stored
- __--output_type, -o__: to select the format of the CC block prediction. The type outputs allowed are: .csv (default), .tiff and .asc.
- __--n_cpu, -n__: Number of CPU that will be used (default: -1 -> max cpu available)

Given the input path and the output type, the script will provide the CC prediction for each block which will be stored in the same path_to_blocks_data directory.  
__Example__: 
> python3 Main_CC_Prediction.py /lex/shared/projects/nnveget/data/control_test_laz /lex/shared/projects/nnveget/data/control_test_laz -o asc -n 02


### CC Results
For predicting the Canopy Cover, 80% of Calonian blocks were used. This corresponds to arround 2.5 TB of data, although the trained model only uses 200MB. The other 20% blocks were used to test and validate the consistency of the model.

With 4:30h of processing time and 45 min of model training, we achieved the following scored:

- <ins>R-Score</ins>: 0.89 (Train), **0.88 (Test)**

- <ins>MAE</ins>: 6.30 (Train), **6.47 (Test)**

Model does not overfit and for a group of predictions the average absolute error is estimated to be of 6.47%.

The following images compare the real vs prediction of blocks from different geographical regions that were NOT used on the training.

![cap1](https://github.com/pau-ventrod/GFire-CC-prediction/assets/105445981/e80c112b-7621-4f1b-b0f0-9dac07671353)

![cap2](https://github.com/pau-ventrod/GFire-CC-prediction/assets/105445981/2e1103bb-cf94-4aab-881c-6b2ff352299e)

The following image shows the comparison between all the catalonian CC and the prediction that our model provided.

![cap3](https://github.com/pau-ventrod/GFire-CC-prediction/assets/105445981/8582127c-d849-469e-bdd3-fba66ed48e69)

On a higher scale, the errors are much harder to spot.

## Error analysis

The following confusion matrix shows the percentage of classified points according to the predicted versus real values. The optimal situation would be a completely diagonal matrix.

![WhatsApp Image 2023-10-30 at 17 03 05](https://github.com/GFire-UAB/CC-Prediction/assets/105445981/33ea5c01-47e4-47bd-88d6-b9af9efc1eed)

As we can see, most of the predictions fall into the matrix's diagonal and near it, with a clear tencency to underestimate, which is good as our groundtruth came from an overestimation of the reality. Note aswell that 56% of the classifications are well-classified into a low range (CC belonging to 0-10%), which might aswell result in the model tending to underestimate.

Analysing the absolute valued error of our predictions:

![WhatsApp Image 2023-10-30 at 18 12 30](https://github.com/GFire-UAB/CC-Prediction/assets/105445981/81f92f5b-e760-45fb-8562-9810c24b7e24)

More than 62% of the predictions have a very low error (<5%), while the error is <20% in more than the 90% of the cases. Beyond the 95% of the cases the error will be upper bounded by 30%.

## Script execution time

A batch of tests have been perfomed in order to compare how did the processing behave on a different density of points. 

Three LiDAR sources have been used:
- Navarra: With a minimum point density of 10 points/m^2, and blocks of 1km x 1km.
- Canada: With a minimum point density of 6 points/m^2, and blocks of 1km x 1km.
- Catalonia: With a minimum point density of 0.5 points/m^2, and blocks of 2km x 2km.

Different steps of the process have been monitored, and we found out that the only parts that had a significant execution time were block preprocessing and tiff preprocessing. 
The executions have been made using a different number of CPU to check how did the model perform under paralelization, and also using a different number of blocks.

The following graph shows how the main bulk of the program (block preprocessing time) scales as the size of the input data increases, and how the paralelization can help mitigate this factor. The graph also provides time execution estimations for real case examples.

![Block_Execution](https://github.com/pau-ventrod/GFire-CC-prediction/assets/105445981/4aa9f00a-1e0d-4597-b252-246910e8b895)

We have noticed that, as input size grows, tiff preprocessing remains constant, which gives all the concernedness of time execution to the block preprocessing part. Note also that paralelizing has a huge impact to final executing time, decreasing it from 20 minutes to 3 minutes by using 9 CPUs for an input data consisting of 9 blocks with 10 points/m^2 density.

### Members of the team:

- Alejandro Donaire: aledonairesa@gmail.com | https://www.linkedin.com/in/alejandro-donaire

- Èric Sánchez: ericsanlopez@gmail.com | www.linkedin.com/in/ericsanlopez

- Pau Ventura: pau.ventura.rodriguez@gmail.com | https://www.linkedin.com/in/pauvr

- Francesco Tedesco francescotedesco7d2@gmail.com | https://www.linkedin.com/in/francescotedesco7d2/
