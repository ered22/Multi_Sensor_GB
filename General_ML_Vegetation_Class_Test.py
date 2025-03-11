#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 06:53:24 2024

@author: eoinreddin
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn import metrics
from itertools import compress 
from scipy import stats
import shap
import random
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Calculate Feature Importances
def calc_importances(model,X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    SortedNames = [list(X)[idx] for idx in indices]
    return(importances,indices,SortedNames)

# Cross Validation
def CrossVal_Func(model,score_met,cv,Model_String,Score_String):
    scores = cross_val_score(model, X, y, scoring=score_met, cv=cv,n_jobs=3) # two core parallel processing significantly improves runtime
    print('\n' + Model_String + ' [Min, Med, Mean, Max] Cross Validated ' + Score_String + ": [" + str(round(min(scores),2))+ ", " + str(round(np.median(scores),2))  
          + ", " + str(round(np.mean(scores),2)) + ", " + str(round(max(scores),2)) + "]")
    return

#%%
Drop_Deep = "Y"
Met_Filter = "Y"

Veg_Filter = "N"
Veg_or_Bare = "Bare" # which to perform the modelling on ("Veg" or "Bare")

Test_Bunds = "N" # To check for bunds or not 
Keep_Bunds = "N" # To output the bunded bogs or non-bunded
                # Only works if test bunds if set to "Y"
Perform_RFE = "Y"
N_Features = 11 # For Bare Peat with Bunds, use 12; Bare peat without bunds: 11
                #for all bare peat use 15
                # For vegetation, use 8; for all data use 13
                
Depth_Filter = "N"

Include_PD = "N"

FilterSeason = "N"
# Either SS or AW
Season2Filter = "SS"

DetectOutlier = "N"

# Perform Deep learning?
Do_DL = "N"

rs = 42
ts = 0.2
kfolds = 5

#%% Read Data
GurteenCSV = pd.read_csv('dly1475/dly1475.csv',skiprows=24)

GurteenCSV.date = pd.to_datetime(GurteenCSV.date, format="%d-%b-%Y")

PZ_W_PD = pd.read_csv('Piez_w_PD.csv')
PZ_W_PD['Full_Well_ID'] = PZ_W_PD.SiteName + '_' + PZ_W_PD.WellID

with open('General_Model_Data_v2.pkl', 'rb') as file1:
    DF_4_RF = pickle.load(file1)

# Random Wells for input in classifier
print(DF_4_RF.Well_ID.drop_duplicates().sample(30,random_state = 42,ignore_index = True))
#%% Landcover classes
# State vegetated and bare peat bogs for filtering
#Bare_Peat = ["Ballaghurt", "Begnagh", "Bloomhill", "Bracklin West", "Bunahinly", "Carranstown",
#             "Castlegar", "Clooneeny","Clooniff", "Clynan", "Daingean Derries", "Derrinboy",
#             "Derryfadda", "Derrymoylin", "Edera","Esker", "Gilltown", "Glebe", "Glenlough", 
#             "Granaghan", "Kilashee","Kilgarvin", "Killaranny","Lemanaghan West",
#             "Lisclogher West","Lodge","Mostrim", "Mouds", "Newtown/Loughgore", "Pollagh",
#             "Prosperous", "Tirrur Derrymore","Tonalig","Torr","Ummeras"]

Bare_Peat = ["Ballaghurt_008_S", "Ballaghurt_017_S", "Begnagh_002_S", "Begnagh_006_S", 
             "Begnagh_012_S", "Belmount_006_S", "Belmount_015_S", "Blackwater_002_S", 
             "Blackwater_006_S", "Blackwater_008_S", "Blackwater_042_S",
             "Blackwater_046_S", "Blackwater_049_S", "Blackwater_053_S", "Bloomhill_004_S", 
             "Bloomhill_005_S", "Bloomhill_009_S", "Bloomhill_010_S", "Bloomhill_013_S", 
             "Bloomhill_015_S", "Bloomhill_018_S", "Bracklin West_003_S",
             "Bracklin West_004_S", "Carranstown_001_S", "Carranstown_004_S", 
             "Carranstown_008_S", "Castlegar_001_S", "Castlegar_002_S", "Castlegar_003_S", 
             "Castlegar_004_S", "Castlegar_005_S", "Castlegar_006_S", "Castlegar_007_S", 
             "Castlegar_008_S", "Castlegar_009_S", "Castlegar_010_S", "Castlegar_011_S", 
             "Castlegar_012_S", "Castlegar_013_S", "Castlegar_014_S", "Castlegar_015_S", 
             "Castlegar_016_S", "Castlegar_017_S", "Castlegar_018_S", "Castlegar_019_S", 
             "Castlegar_020_S", "Castlegar_021_S", "Castlegar_022_S", "Castlegar_023_S", 
             "Castlegar_024_S", "Castlegar_025_S", "Castlegar_026_S", "Castlegar_027_S", 
             "Castlegar_028_S", "Castlegar_029_S", "Castlegar_030_S", "Castlegar_032_S",
             "Castlegar_033_S", "Castlegar_034_S", "Castlegar_035_S", "Castlegar_036_S", 
             "Castlegar_037_S", "Castlegar_038_S", "Castlegar_039_S", "Cavemount_002_S", 
             "Clonad_003_S", "Clonad_005_S", "Clonad_007_S", "Clonad_022_S", 
             "Clongawney More_011_S", "Clongawney More_017_S", "Clongawney More_020_S",
             "Clooneeny_001_S", "Clooneeny_007_S", "Clooneeny_010_S", "Clooneeny_014_S", 
             "Clooniff_001_S", "Clooniff_004_S", "Clooniff_005_S", "Clooniff_009_S", 
             "Clooniff_014_S", "Cloonshannagh_003_S", "Cloonshannagh_010_S",
             "Cloonshannagh_015_S", "Daingean Derries_005_S", "Daingean Derries_011_S", 
             "Derraghan_001_S", "Derrinboy_003_S", "Derrinboy_005_S", "Derrinboy_008_S", 
             "Derrinboy_012_S", "Derrinboy_014_S", "Derrinboy_015_S", "Derryadd East_003_S", 
             "Derryadd East_009_S", "Derrycashel_003_S", "Derrycashel_017_S", 
             "Derrycashel_019_S", "Derrycolumb_001_S", "Derrycolumb_005_S", "Derrycolumb_011_S", 
             "Derrycolumb_012_S", "Derrycolumb_014_S", "Derrycolumb_017_S", "Derrycolumb_018_S",
             "Derrycolumb_020_S", "Derryfadda_002_S", "Derryfadda_004_S", "Derryfadda_013_S", 
             "Derryfadda_017_S", "Derryfadda_021_S", "Derryshannoge_012_S", "Drinagh Phase 2_014_S", 
             "Edera_001_S", "Edera_002_S", "Edera_006_S", "Edera_009_S", "Edera_011_S", "Esker_002_S", 
             "Esker_004_S", "Esker_005_S", "Esker_011_S", "Esker_013_S", "Esker_024_S", 
             "Garryduff_001_S", "Garryduff_002_S", "Garryduff_019_S", "Gilltown_001_S", 
             "Gilltown_007_S", "Gilltown_010_S", "Gilltown_012_S", "Gilltown_014_S", 
             "Gilltown_015_S", "Granaghan_002_S", "Killaranny_002_S", "Killaranny_004_S",
             "Killaranny_010_S", "Kilmacshane_001_S", "Kilmacshane_014_S", "Kilmacshane_015_S",
             "Kilmacshane_021_S", "Kilmacshane_023_S", "Kilmacshane_026_S", "Kilmacshane_029_S",
             "Lodge_010_S", "Lodge_011_S", "Lodge_013_S", "Mount Lucas_001_S", "Mount Lucas_005_S",
             "Mount Lucas_007_S", "Mount Lucas_008_S", "Mount Lucas_019_S", "Mount Lucas_022_S",
             "Mount Lucas_024_S", "Mount Lucas_027_S", "Oughter_004_S", "Oughter_014_S", 
             "Oughter_018_S", "Pollagh_005_S", "Pollagh_009_S", "Pollagh_011_S", "Pollagh_012_S", 
             "Torr_001_S", "Torr_004_S", "Torr_007_S", "Torr_010_S", "Torr_012_S", "Torr_015_S", 
             "Turraun_004_S", "Ummeras_009_S", "Ummeras_011_S"]


Bunds = ["Begnagh_002_S", "Begnagh_006_S", "Begnagh_012_S", "Belmount_006_S",
         "Bloomhill_004_S", "Bloomhill_005_S", "Bloomhill_018_S", "Bracklin West_003_S",
         "Carranstown_001_S", "Carranstown_004_S", "Carranstown_008_S", "Castlegar_009_S",
         "Castlegar_010_S", "Castlegar_011_S", "Castlegar_012_S", "Castlegar_013_S",
         "Castlegar_021_S", "Castlegar_023_S", "Castlegar_024_S", "Castlegar_025_S",
         "Castlegar_026_S", "Castlegar_033_S", "Castlegar_034_S", "Castlegar_035_S",
         "Clonad_003_S", "Clonad_007_S", "Clooneeny_001_S", "Clooneeny_007_S",
         "Clooneeny_010_S", "Clooneeny_014_S", "Clooniff_001_S", "Clooniff_004_S",
         "Clooniff_005_S", "Clooniff_009_S", "Clooniff_014_S", "Daingean Derries_005_S", 
         "Daingean Derries_011_S", "Derraghan_001_S", "Derrinboy_003_S", "Derrinboy_005_S",
         "Derrinboy_008_S", "Derrinboy_012_S", "Derrinboy_014_S", "Derrinboy_015_S",
         "Derrycolumb_011_S", "Derrycolumb_017_S", "Derrycolumb_018_S", "Edera_006_S",
         "Esker_004_S", "Esker_013_S", "Garryduff_001_S", "Killaranny_002_S",
         "Killaranny_004_S", "Kilmacshane_021_S", "Lodge_011_S", "Lodge_013_S",
         "Mount Lucas_001_S", "Mount Lucas_007_S", "Mount Lucas_008_S", "Pollagh_009_S",
         "Pollagh_011_S", "Pollagh_012_S", "Ummeras_009_S", "Ummeras_011_S"]



#Bunds = ["Ballaghurt","Begnagh","Belmount","Carranstown","Castlegar","Clooneeny","Daingean Derries",
#         "Killaranny","Lodge","Pollagh","Prosperous","Ummeras"]

#%% Filter Data
# Drop deep wells from dataframe
if Drop_Deep == "Y":
        DF_4_RF = DF_4_RF[~DF_4_RF.Well_ID.str.contains("_D")]

if Depth_Filter == "Y":
    DF_4_RF = DF_4_RF.drop(DF_4_RF[DF_4_RF.WTD > 50].index)

# Clean data to exclude values outside of -1 and 1
DF_4_RF = DF_4_RF[DF_4_RF.NDVI < 1]
# Clean data to exclude values outside of -1 and 1
DF_4_RF = DF_4_RF[DF_4_RF.NDVI > -1]
# Clean data to exclude red values less than 0 
DF_4_RF = DF_4_RF[DF_4_RF.Red >= 0]
# Clean data to exclude NDWI values less than -1 
DF_4_RF = DF_4_RF[DF_4_RF.NDWI >= -1]
# Clean data to exclude NDWI values less than -1 
DF_4_RF = DF_4_RF[DF_4_RF.STR <= 40]

if Met_Filter == "Y":
    # Convert to datetime
    DF_4_RF['Date'] = pd.to_datetime(DF_4_RF['Date']).dt.date
    DF_4_RF['Month'] = pd.to_datetime(DF_4_RF.Date).dt.month_name()
    GurteenCSV['date'] = pd.to_datetime(GurteenCSV['date']).dt.date
    # Find meterological parameters for each date
    DF_4_RF['rain']  = pd.to_datetime(DF_4_RF.Date).map(GurteenCSV.set_index(GurteenCSV.date)['rain']).astype(float)
    DF_4_RF['soil']  = DF_4_RF.Date.map(GurteenCSV.set_index(GurteenCSV.date)['soil']).astype(float)
    #DF_4_RF['MaxT']  = DF_4_RF.Date.map(GurteenCSV.set_index(GurteenCSV.date)['maxtp']).astype(float)
    DF_4_RF['MinT']  = DF_4_RF.Date.map(GurteenCSV.set_index(GurteenCSV.date)['mintp']).astype(float)
   # DF_4_RF['evap']  = DF_4_RF.Date.map(GurteenCSV.set_index(GurteenCSV.date)['evap']).astype(float)
    #DF_4_RF['wdsp']  = DF_4_RF.Date.map(GurteenCSV.set_index(GurteenCSV.date)['wdsp']).astype(float)
    #DF_4_RF['smd_md']  = DF_4_RF.Date.map(GurteenCSV.set_index(GurteenCSV.date)['smd_md']).astype(float)
    #DF_4_RF['ddhm']  = DF_4_RF.Date.map(GurteenCSV.set_index(GurteenCSV.date)['ddhm']).astype(float)

    # Drop based on value    
    DF_4_RF = DF_4_RF.drop(DF_4_RF[DF_4_RF.rain > 7.5].index)
    DF_4_RF = DF_4_RF.drop(DF_4_RF[DF_4_RF['MinT'] < 2].index)
    #DF_4_RF.drop(['soil','rain'], inplace=True, axis=1) 

#%% Test snippet to see about auto classification
# #%% Short test to see about classifying landcover types before splitting them
# LC_Class = pd.DataFrame(
#     {"Vegetation": ["Blackwater_027_S","Clonad_002_S","Cloncreen_015_S","Cloncreen_025_S","Derries_014_S","Derrybrat_004_S",
#                     "Drinagh_007_S","Esker_019_S","Killaranny_006_S","Timahoe North_010_S"],
#      "Bare Peat": ["Blackwater_002_S","Castlegar_008_S","Cloncreen_033_S","Clynan_002_S","Daingean Derries_005_S",
#                    "Gilltown_007_S","Glebe_003_S","Knappoge_013_S","Timahoe South_026_S","Timahoe South_043_S"],
#      "Bunds": ["Bloomhill_009_S","Boora Bog_006_S","Bracklin West_004_S","Clonad_007_S","Clooneeny_001_S","Daingean Derries_011_S",
#                "Derrycolumb_017_S","Derryfadda_002_S","Esker_004_S","Lodge_011_S"]
    
#     }
#     )

# DF_4_Class = DF_4_RF.copy()
# DF_4_Class["Landcover"] = "Default"
# for idx,val in enumerate(LC_Class.Vegetation):
#     DF_4_Class["Landcover"][DF_4_Class.Well_ID == LC_Class["Vegetation"][idx]] = 'Vegetation'
#     DF_4_Class["Landcover"][DF_4_Class.Well_ID == LC_Class["Bare Peat"][idx]] = 'Bare Peat'    
#     DF_4_Class["Landcover"][DF_4_Class.Well_ID == LC_Class["Bunds"][idx]] = 'Bunds'


# DF_4_Class = DF_4_Class[DF_4_Class.Landcover != "Default"]

# # Separate features and target variable for Machine Learning
# X = DF_4_Class.drop(['WTD','Well_ID','Date','Month','rain','soil','MinT','Landcover'], axis=1)
# y = DF_4_Class['Landcover']

# # Split the data into training and testing sets for later
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=rs)

# # Set up random forest model
# modelRF = RandomForestClassifier(n_estimators=500, criterion='gini', bootstrap=True,
#                                  oob_score=True,random_state=42,max_depth=5, min_samples_split=2)

# # Train the Random Forest model
# modelRF.fit(X_train, y_train)
# # Predict on the test set
# y_pred = modelRF.predict(X_test)

# # Results: using metrics module for accuracy calculation
# print("\nACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))
# print("\nCONFUSION MATRIX:\n", metrics.confusion_matrix(y_test, y_pred))
# print("\nCLASSIFICATION REPORT:\n", metrics.classification_report(y_test, y_pred))

# DF_4_Landcover = DF_4_RF.copy()
# X = DF_4_Class.drop(['WTD','Well_ID','Date','Month','rain','soil','MinT','Landcover'], axis=1)

# Landcover = modelRF.predict(X_test)

# if FilterSeason == "Y":
#     if Season2Filter == "SS":    
#         Months2Keep = ["April","May","June","July","August","September"]
#     elif Season2Filter == "AW":  
#         Months2Keep = ["October","November","December","January","March"]  
#     DF_4_RF = DF_4_RF[DF_4_RF.Month.isin(Months2Keep)]


# #%% Subset full dataset by bare peat, bunded, and vegetated
# # Filter and drop based on whether there are bunds present or not
# if Veg_Filter == "Y":
#     print('\nPerforming Vegetation Filtering')
#     DF_4_RF['Landcover'] = Landcover
#     # If bare peat
#     if Veg_or_Bare == "Bare":
#         DF_4_RF = DF_4_RF[DF_4_RF.Landcover != 'Vegetation']
#     # Option to filter based on bunds or not
#         if Test_Bunds == "Y":
#             if Keep_Bunds == "Y":
#                 DF_4_RF = DF_4_RF[DF_4_RF.Landcover == 'Bunds']
#             elif Keep_Bunds == "N":   
#                 DF_4_RF = DF_4_RF[DF_4_RF.Landcover != 'Bunds']
#     # If vegetation        
#     elif Veg_or_Bare == "Veg":
#         DF_4_RF = DF_4_RF[DF_4_RF.Landcover == 'Vegetation']
#     # If error 
#     else:
#         print('\nVariable <Veg_or_Bare> must be assigned either "Veg" or "Bare". Exiting script.')
#         sys.exit()
    
#     # Remove Landcover option
#     DF_4_RF = DF_4_RF.drop('Landcover', axis=1)


#%% NDVI Test
beans = [0]*len(DF_4_RF.Well_ID.unique())
for idx,val in enumerate(DF_4_RF.Well_ID.unique()):
    mean2place = DF_4_RF.EVI[DF_4_RF.Well_ID == val].mean()
    beans[idx] = mean2place
plt.hist(beans,bins = 200)
#%% Subset full dataset by bare peat, bunded, and vegetated
# Filter and drop based on whether there are bunds present or not
if Veg_Filter == "Y":
    print('\nPerforming Vegetation Filtering')
    Bare_2_Append = []
    for [idx,val] in enumerate(Bare_Peat):
        if Veg_or_Bare == "Bare":
            Bare_Vals = DF_4_RF[DF_4_RF.Well_ID.str.contains(val)]      
            Bare_2_Append.append(Bare_Vals)  
        elif Veg_or_Bare == "Veg":
            DF_4_RF = DF_4_RF[~DF_4_RF.Well_ID.str.contains(val)]
        else:
            print('\nVariable <Veg_or_Bare> must be assigned either "Veg" or "Bare". Exiting script.')
            sys.exit()
    if Veg_or_Bare == "Bare":
        DF_4_RF = pd.concat(Bare_2_Append)

# Filter and drop based on wheter bunds are present or not
if Veg_Filter == "Y" and Test_Bunds == "Y" and Veg_or_Bare == "Bare":
    Bund_2_Append = []
    for [idx,val] in enumerate(Bunds):
        if Keep_Bunds == "Y":
            Bund_Vals = DF_4_RF[DF_4_RF.Well_ID.str.contains(val)]      
            Bund_2_Append.append(Bund_Vals)
        elif Keep_Bunds == "N":
            DF_4_RF = DF_4_RF[~DF_4_RF.Well_ID.str.contains(val)]
    if Keep_Bunds == "Y":
        DF_4_RF = pd.concat(Bund_2_Append)        

#DF_4_RF = DF_4_RF[DF_4_RF.NDVI > 0.55]

if Include_PD == "Y":
    DF_4_RF.reset_index(inplace = True,drop = True)
    DF_4_RF['Peat_Depth'] = float(np.nan)
    for [idx,val] in enumerate(DF_4_RF.Well_ID):
        PD_Val = PZ_W_PD.SAMPLE_1[PZ_W_PD.Full_Well_ID == val]
        if len(PD_Val) > 0:
            DF_4_RF['Peat_Depth'][idx] =  PD_Val
    DF_4_RF = DF_4_RF.dropna(how='any')    
    #DF_4_RF = DF_4_RF[DF_4_RF.Peat_Depth != 0]    
    PZ_2_Out = PZ_W_PD[PZ_W_PD['Full_Well_ID'].isin(list(DF_4_RF.Well_ID.unique()))]
    
if FilterSeason == "Y":
    if Season2Filter == "SS":    
        Months2Keep = ["April","May","June","July","August","September"]
    elif Season2Filter == "AW":  
        Months2Keep = ["October","November","December","January","March"]  
    DF_4_RF = DF_4_RF[DF_4_RF.Month.isin(Months2Keep)]
  
#%% Format training and testing dataframes        
# Separate features and target variable for Machine Learning
X = DF_4_RF.drop(['WTD','Well_ID','Date','Month'], axis=1)
y = DF_4_RF['WTD']

if DetectOutlier == "Y":
    OutlierInds = (np.abs(stats.zscore(X)) < 3).all(axis=1)
    X = X[OutlierInds]
    y = y[OutlierInds]

# Split the data into training and testing sets for later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts,random_state=rs)

# Random Forest: use k-fold CV to evaluate RF model on full dataset
modelRF = RandomForestRegressor(n_estimators=500, criterion='squared_error', bootstrap=True,
                              oob_score=True,random_state=rs,max_depth=50, min_samples_split=2,n_jobs=2)
# Gradient Boost: use k-fold CV to evaluate GB model on full dataset
modelGB = GradientBoostingRegressor(n_estimators = 500, max_depth = None, max_leaf_nodes = 12,
    random_state = rs, learning_rate = 0.01)

#%% Recursive Feature Selection
if Perform_RFE == "Y":
    selectorGB = RFE(modelGB, n_features_to_select=N_Features, step=1)
    selectorRF = RFE(modelRF, n_features_to_select=N_Features, step=1)
    selectorGB = selectorGB.fit(X, y)
    selectorRF = selectorRF.fit(X, y)
    GB_Features = selectorGB.support_
    RF_Features = selectorRF.support_
    
    Features = list(X)

    GB_Features = list(compress(Features, GB_Features))
    RF_Features = list(compress(Features, RF_Features))

    X_GB = X[GB_Features]
    X_test_GB = X_test[GB_Features]
    X_train_GB = X_train[GB_Features]
    
    X_RF = X[RF_Features]
    X_test_RF = X_test[RF_Features]
    X_train_RF = X_train[RF_Features]
else:
    X_GB = X.copy() 
    X_test_GB = X_test.copy()
    X_train_GB = X_train.copy()
    
    X_RF = X.copy() 
    X_test_RF = X_test.copy()
    X_train_RF = X_train.copy()

#%% Train the Random Forest model
modelRF.fit(X_train_RF, y_train)
# Predict on the test set
y_pred = modelRF.predict(X_test_RF)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred) # Use this indexing to only look at range between -40 and 0:[(y_test > -40) & (y_test < 0)]
r2 = r2_score(y_test, y_pred)
print('\n' + 'Random Forest' + f' OOB RMSE: {round(np.sqrt(mse),2)}')
print('Random Forest' + f' OOB R^2 Score: {round(r2,2)}')
[importances,indices,SortedNames] = calc_importances(modelRF,X_RF)
# Linear Regression for best Fit
coef = np.polyfit(y_test,y_pred,1)
poly1d = np.poly1d(coef) 

# Train the Gradient Boosting model
modelGB.fit(X_train_GB, y_train)
# Predict on the test set
y_pred = modelGB.predict(X_test_GB)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred) # Use this indexing to only look at range between -40 and 0:[(y_test > -40) & (y_test < 0)]
r2 = r2_score(y_test, y_pred)
print('\n' + 'Gradient Boosting' + f' OOB RMSE: {round(np.sqrt(mse),2)}')
print( 'Gradient Boosting' + f' OOB R^2 Score: {round(r2,2)}')

# Linear Regression for best Fit
coef = np.polyfit(y_test,y_pred,1)
poly1d = np.poly1d(coef) 

cv = KFold(n_splits=kfolds, shuffle=True,random_state = rs)
CrossVal_Func(modelGB,'r2',cv,'Gradient Boost','r^2')


# Consider SHAP Values
explainer = shap.Explainer(modelGB)
shap_values = explainer(X_train_GB)

plt.figure(figsize=(6, 4))
shap.plots.beeswarm(shap_values, max_display=20,show = False)
plt.title("Beeswarm Plot: Gradient Boosting" )
plt.tight_layout(rect=[0, 0, 1, 0.96])

#%% Figures
fig = plt.figure(constrained_layout=True, figsize=(7.5, 10))
gs = GridSpec(3, 1, figure=fig)
ax1, ax2, ax3= fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])

# Random Forest OOB Scatter Plot
ax1.plot(y_test,y_pred, '.', y_test, poly1d(y_test), '-k',mfc = '#0072B2',mec = 'k',mew = 0.2)
ax1.set_ylabel('Predicted (cm)')

ax1.set_xlabel("Actual (cm)")
ax1.set_title("GB" + ":\nWTD Predicted vs Actual (OOB)")
ax1.annotate('$R^2$: ' + str(round(r2,2)) + "\nRMSE: " + str(round(np.sqrt(mse),2)), xy = (0.7,0.1),xycoords = 'axes fraction')
    
facecols = ['#0072B2','#D55E00','#0072B2','#D55E00']

[importances,indices,SortedNames] = calc_importances(modelGB,X_GB)
ax2.set_xlim([-1, X_GB.shape[1]])
ax2.set_xticks(range(X_GB.shape[1]), SortedNames,rotation=270)
ax2.bar(range( X_GB.shape[1]), importances[indices], align="center",fc = '#0072B2',ec = 'k')
ax2.set_title("GB Feature Importances")

[importances,indices,SortedNames] = calc_importances(modelRF,X_RF)
ax3.set_xlim([-1,  X_RF.shape[1]])
ax3.set_xticks(range(X_RF.shape[1]), SortedNames,rotation=270)
ax3.bar(range(X_RF.shape[1]), importances[indices], align="center",fc = '#0072B2',ec = 'k')
ax3.set_title("RF Feature Importances")

plt.savefig('RF_Results.pdf', dpi = 300)



#%% Deep learning!
if Do_DL == "Y":    
    #MLPREGRESSOR
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    # apply same transformation to test data
    X_test = scaler.transform(X_test) 

    modelMLPR = MLPRegressor(random_state=42, max_iter=2000,hidden_layer_sizes = (1000,),
                             n_iter_no_change=100)
    modelMLPR.fit(X_train, y_train)
    modelMLPR.predict(X_test)
    print(modelMLPR.score(X_test, y_test))

#%% Use RFE with cross-validation to   (from https://www.geeksforgeeks.org/recursive-feature-elimination-with-cross-validation-in-scikit-learn/)
# # find the optimal number of features 
# selector = RFECV(modelGB, cv=kfolds) 
# selector = selector.fit(X, y) 
  
# # Print the optimal number of features 
# print("Optimal number of features: %d" % selector.n_features_) 
  
# # Print the selected features 
# print("Selected features: %s" % selector.support_) 

