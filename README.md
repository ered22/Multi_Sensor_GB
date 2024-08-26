# WETPEAT-Remote_Sensing
- Code maintained on github, contact eoin.reddin@universityofgalway.ie for help
*********************************
Source code for performing Machine Learning analysis, using Random Forest and Gradient Boosting techniques. (Currently script and directory say Random Forest, as this was the first model considered. This may be changed in later updates).

Code is written in Python, using data processed in SNAP, and is fully open source.

*********************************
## The main script is ./Train_Model.py

Main script to perform random forest and gradient boosting modelling on multi-sensor satellite data, and in-situ data.
### Inputs: 
Piezometer data, containing (extra bands calculated in QGIS): 
    -Well_ID,   xcoord,     ycoord,     Distance2Bund,      Distance2Dam,   Peat_Depth

Met Data from Athenry and Mt. Dillon stations, taken from historical met data:
    - Contains a lot of data, but we use: 
            - Air Temperature,      Rainfall,       Soil Temperature
            
Bord Na Mona water table data, for each piezometer location.
    - Here, we average water table data over the entire date contained here, in order to account
    for offsets in acquisition times between S1 and S2 (same day acquisitions)
    
Multi-sensor data in bigtiff format. Calculated, collocated and exported in SNAP. Each level of bigtiff
contains the following indices, at 30 m pixel spacing:
    - VV,       NDVI,       NDWI,       STR,       SAVI,       Cloud Mask,       TCG,
      TWC,      B2,         B3,         B4,        B8,         B11,              B12,         VH
These can be filtered out in introduction

Validated using both cross validation and out-of-bag validation.

### Calculates:
Random Forest Cross Validated r^2;	Random Forest Cross Validated RMSE;
Gradient Boost Cross Validated r^2: 	Gradient Boost Cross Validated RMSE;

Random Forest OOB RMSE;			Random Forest OOB R^2 Score;
Gradient Boosted OOB RMSE;		Gradient Boosted OOB R^2 Score.

### Outputs:
all_models.pkl
     - File containing the Gradient boosting model for prediction intervals
GB_Model.pkl
     - File containing the trained gradient boosting model
Figures: 
Scatter plots of Actual vs Predicted Water Table Depth values, and histograms of Parameter Signifcance, for both Random Forest and Gradient Boosting techniques. **Note: These plots are simply indicative of model performance, as they use the results of OOB validation for ease of presentation.**

*********************************
## The main script is ./Make_Predictions.py

Makes predictions of wtd at a given site

### Inputs: 
all_models.pkl
     - File containing the Gradient boosting model for prediction intervals

GB_Model.pkl
     - File containing the trained gradient boosting model
Both of these are saved in the Train_Model.py script

Multi-sensor data in bigtiff format. Calculated, collocated and exported in SNAP. Each level of bigtiff
contains the following indices, at 30 m pixel spacing ** For the dates to be predicted**:
    - VV,       NDVI,       NDWI,       STR,       SAVI,       Cloud Mask,       TCG,
      TWC,      B2,         B3,         B4,        B8,         B11,              B12,         VH


### Calculates:
Predictions of WTD, prediction intervals for predictions

### Outputs:
Figures: 
Scatter plots of predicted Water Table Depth values for Gradient Boosting model.


*********************************
## Function 1: ./RF_Model.py: Geotiff_2_RF_Array.py

Function to read and format geotifs into something that can be input into RF algorithm
Main steps:
> Read satellite data
> Read Complementary Data
> Format into dataframe for each date. 
> Perform cloud masking

# Inputs
*Geotiff_2_RF_Array(ADates, DDates, Filters, cols2drop, Days2AVG, FilterEnv,
                       WTDThresholds, pathlist, Band_Names):*
**ADates**: List of dates (names of bigtiff files) to be imported, in Ascending direction
**DDates**: List of dates (names of bigtiff files) to be imported, in Descending direction 
**Filters**: List containing flags on whether to filter data by columns of optical data, and meteorological data
**cols2drop**: Names of columns to drop if flag is activated in *Filters*
**Days2AVG**: Number of days to average meteorological data across (e.g. before acquisition). Rational here, for example, is that soil moisture will be controlled by rainfall in the days up to the acquisition, and not just on the day of acquisition
**FilterEnv**: Flag giving option to provide limits on water table depth. (e.g. exclude those outside of range)
**WTDThresholds**: Lower and upper limits of water table depth to consider if *FilterEnv* is activated
**Pathlist**: List of paths pointing to all the data that has to be loaded
**Band_Names**: List containing the names of bands (in correct order) contained in the bigtiff outputs from SNAP.

# Outputs:
Returns variable *DF_4_RF*. A dataframe containing the values to input into ML modelling (where each line corresponds to the relevant variable at piezometer)
Column headers: VV, NDVI, NDWI, STR, SAVI, CloudMask, TCG, TCW, Blue, Green, Red, NIR, SWIR1, SWIR2, VH, AVG_WTD, Distance2Bund, Distance2Dam, PeatDepth
Final column headers have variables indicated by *cols2drop* removed


*********************************
## Function 1: ./RF_Model.py: Geotiff_2_RF_Predict.py

Function to read and format geotifs into something that can be input into RF algorithm to predict WTD
*Very similar to Geotiff_2_RF_Array, except values are calculated at every pixel in the bog (not just piezometers), and AVG_WTD is not included as a column, as we are trying to predict it from this dataset.

# Inputs
*Geotiff_2_RF_Array(ADates, Filters, cols2drop, Days2AVG, FilterEnv,
                       WTDThresholds, pathlist, Band_Names):*
**ADates**: List of dates (names of bigtiff files) to be imported, in Ascending direction (DDates not included as we only consider single date)
**Filters**: List containing flags on whether to filter data by columns of optical data, and meteorological data
**cols2drop**: Names of columns to drop if flag is activated in *Filters*
**Days2AVG**: Number of days to average meteorological data across (e.g. before acquisition). Rational here, for example, is that soil moisture will be controlled by rainfall in the days up to the acquisition, and not just on the day of acquisition
**FilterEnv**: Flag giving option to provide limits on water table depth. (e.g. exclude those outside of range)
**WTDThresholds**: Lower and upper limits of water table depth to consider if *FilterEnv* is activated
**Pathlist**: List of paths pointing to all the data that has to be loaded
**Band_Names**: List containing the names of bands (in correct order) contained in the bigtiff outputs from SNAP.

# Outputs:
Returns variable *DF_4_RF*. A dataframe containing the values to input into ML modelling for predicting WTD across the entire bog
Column headers: VV, NDVI, NDWI, STR, SAVI, CloudMask, TCG, TCW, Blue, Green, Red, NIR, SWIR1, SWIR2, VH, Distance2Bund, Distance2Dam, PeatDepth
Final column headers have variables indicated by *cols2drop* removed


*********************************
## *.xml files
Processing graphs for SNAP processing

