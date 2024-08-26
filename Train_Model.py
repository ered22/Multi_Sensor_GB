#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:14:40 2024 @author: eoin.reddin@universityofgalway.ie
For detailed description refer to README.md
"""
# Import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.gridspec import GridSpec
from Geotiff_2_RF_Array import Geotiff_2_RF_Array
import pickle

#%% Relevant Functions
# Function to set up paths
def prep_txt_files(bogname,asc_dir,desc_dir,PZString): 
    TopPath = "/Users/eoinreddin/Library/CloudStorage/OneDrive-NationalUniversityofIreland,Galway/Galway_Post_Doc/"
    path2Asc = TopPath + "WETPEAT-Remote_Sensing/Random_Forest/Colocated_Tifs/" + bogname +"/" + asc_dir + "/"
    path2desc = TopPath + "WETPEAT-Remote_Sensing/Random_Forest/Colocated_Tifs/" + bogname +"/" + desc_dir + "/"
    path2piezo = TopPath + "WETPEAT-Remote_Sensing/Piezometer_Data/" + bogname + PZString + ".csv"
    path2wtd = TopPath + "WETPEAT-Remote_Sensing/BnM_WTD_Data/" + bogname +"WLC.csv"
    path2athenry = TopPath + "WETPEAT-Remote_Sensing/Random_Forest/Met_Data/Athenry/dly1875.csv"
    path2mtdillon = TopPath + "WETPEAT-Remote_Sensing/Random_Forest/Met_Data/MtDillon/dly1975.csv"
    pathlist = [path2Asc,path2desc,path2piezo, path2wtd, path2athenry, path2mtdillon]
    pathlist = [path2Asc,path2desc,path2piezo, path2wtd, path2athenry, path2mtdillon]
    return(pathlist)

# Cross Validation
def CrossVal_Func(model,score_met,cv,Model_String,Score_String):
    scores = cross_val_score(model, X, y, scoring=score_met, cv=cv,n_jobs=3) # two core parallel processing significantly improves runtime
    print('\n' + Model_String + ' [Min, Med, Mean, Max] Cross Validated ' + Score_String + ": [" + str(round(min(scores),2))+ ", " + str(round(np.median(scores),2))  
          + ", " + str(round(np.mean(scores),2)) + ", " + str(round(max(scores),2)) + "]")
    return

# Fit Models
def fit_models(model,X_train,y_train,X_test,y_test,modelstring):
    # Train the Random Forest model
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred) # Use this indexing to only look at range between -40 and 0:[(y_test > -40) & (y_test < 0)]
    r2 = r2_score(y_test, y_pred)
    print('\n' + modelstring + f' OOB RMSE: {round(np.sqrt(mse),2)}')
    print( modelstring + f' OOB R^2 Score: {round(r2,2)}')
    [importances,indices,SortedNames] = calc_importances(model)
    # Linear Regression for best Fit
    coef = np.polyfit(y_test,y_pred,1)
    poly1d = np.poly1d(coef) 
    return(importances,indices,SortedNames,y_pred,poly1d,r2,mse)

# Calculate Feature Importances
def calc_importances(model):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    SortedNames = [list(X)[idx] for idx in indices]
    return(importances,indices,SortedNames)

#%% Set up
# Additional Data Filter
FilterEnv = "N"     # Set Minimum and maximum Water Table Depth to drop 
WTDthresholds = [-200, 25]  
Filter2 = "Y"       # Remove Columns if needed
cols2drop = ["VV","TCG","SAVI","Red","Green","NIR","NDVI"]
FilterMet = "Y"     # Include meterological data
# Array for including in function
Filters = [Filter2,FilterMet]

 # Number of days before date to average rainfall
Days2AVG = 4;

ts = 0.33 # Test Size
rs = 42 # For random state(FYI: 42 is used as a joke (hitchhikers guide to galaxy))
kfolds = 5 # Number of folds in cross validation

# Band names for each layer in GEOTIFF **N.B these must match those exported from SNAP**
Band_Names = ["VV","NDVI","NDWI","STR","SAVI","CloudMask","TCG","TCW","Blue","Green","Red","NIR","SWIR1","SWIR2","VH","AVG_WTD","Dist2Bund","Dist2Dam","PeatDepth"]
bogname = ["Castlegar","Clooneeny"]
MLNames = ["Random Forest","Gradient Boosting"]

#%% Import data 
# Run for Castlegar
ADates = ["10-09-2022","17-02-2021","21-11-2021","28-08-2021","13-04-2021","17-07-2021",
          "25-04-2021","29-06-2021","16-11-2022","21-03-2022","26-03-2022" ,"30-05-2021"] # Ascending Dates
DDates = ["03-04-2021","04-12-2021","16-03-2021","22-06-2023","03-05-2021","08-01-2021",
           "16-03-2022","29-12-2022","04-03-2022","15-04-2021","19-07-2021","30-10-2021"] # Descending Dates

pathlist = prep_txt_files(bogname[0],"Zero_Day_A","Zero_Day_D","PZ")
# Prepare data for Castlegar
DF_4_RF = Geotiff_2_RF_Array(ADates, DDates, Filters, cols2drop, Days2AVG, FilterEnv, WTDthresholds, pathlist, Band_Names)

# Run for Clooneeny
ADates = ["10-09-2022","16-11-2022"]#,"17-02-2021","21-11-2021","28-08-2021","13-04-2021", "17-07-2021",
          #"25-04-2021","29-06-2021","16-11-2022","21-03-2022","26-03-2022" ,"30-05-2021"] # Ascending Dates
DDates = ["22-06-2023"]#"03-04-2021","16-03-2021","03-05-2021","15-04-2021","30-10-2021","22-06-2023",] # Descending Dates
pathlist = prep_txt_files(bogname[1],"Zero_Day_A","Zero_Day_D","PZ")
DF_4_RF2 = Geotiff_2_RF_Array(ADates, DDates, Filters, cols2drop, Days2AVG, FilterEnv, WTDthresholds, pathlist, Band_Names)

# Merge dataframes together
DF_4_RF = pd.concat([DF_4_RF,DF_4_RF2])
    
# Separate features and target variable for Machine Learning
X = DF_4_RF.drop('AVG_WTD', axis=1)
y = DF_4_RF['AVG_WTD']
# Split the data into training and testing sets for later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts,random_state=rs)

#%% Test Cross Validation
cv = KFold(n_splits=kfolds, shuffle=True,random_state = rs)
# Random Forest: use k-fold CV to evaluate RF model on full dataset
modelRF = RandomForestRegressor(n_estimators=500, criterion='squared_error', bootstrap=True,
                              oob_score=True,random_state=rs,max_depth=50, min_samples_split=2,n_jobs=2)
CrossVal_Func(modelRF,'r2',cv,'Random Forest','r^2')
CrossVal_Func(modelRF,'neg_root_mean_squared_error',cv,'Random Forest','RMSE')
# Fit model and calculate feature importance
RF_Model =  modelRF.fit(X,y)
[importancesFullRF,indicesFullRF,SortedNamesFullRF] = calc_importances(RF_Model)

# Gradient Boost: use k-fold CV to evaluate GB model on full dataset
modelGB = GradientBoostingRegressor(n_estimators = 500, max_depth = None, max_leaf_nodes = 12,
    random_state = rs, learning_rate = 0.05)
CrossVal_Func(modelGB,'r2',cv,'Gradient Boost','r^2')
CrossVal_Func(modelGB,'neg_root_mean_squared_error',cv,'Gradient Boost','RMSE')
# Fit model and calculate feature importance
GB_Model =  modelGB.fit(X,y)
[importancesFullGB,indicesFullGB,SortedNamesFullGB] = calc_importances(GB_Model)

#%% OOB Testing    
# Random Forest
[importancesRF,indicesRF,SortedNamesRF,y_predRF,poly1d_RF,r2RF,mseRF] = fit_models(modelRF,X_train,y_train,X_test,y_test,"Random Forest")
# Gradient Boosting
[importancesGB,indicesGB,SortedNamesGB,y_predGB,poly1d_GB,r2GB,mseGB] = fit_models(modelGB,X_train,y_train,X_test,y_test,"Gradient Boosted")
# Lists to allow for looping when plotting significance plots
SortedNames = [SortedNamesRF, SortedNamesGB, SortedNamesFullRF, SortedNamesFullGB]
importances = [importancesRF[indicesRF], importancesGB[indicesGB], importancesFullRF[indicesFullRF],importancesFullGB[indicesFullGB]]
importancetitles = ["OOB", "OOB", "Full", "Full"]
OOBr2 = [r2RF, r2GB]
OOBrmse = [np.sqrt(mseRF),np.sqrt(mseGB)]

#%% Make Figures
fig = plt.figure(constrained_layout=True, figsize=(7.5, 8))
gs = GridSpec(3, 2, figure=fig)
ax1, ax2, ax3, ax4, ax5, ax6 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

# Random Forest OOB Scatter Plot
ax1.plot(y_test,y_predRF, '.', y_test, poly1d_RF(y_test), '-k',mfc = '#0072B2',mec = 'k',mew = 0.2)
ax1.set_ylabel('Predicted (cm)')

# Gradient Boosted OOB Scatter Plot
ax2.plot(y_test,y_predGB, '.', y_test, poly1d_GB(y_test), '-k',mfc = '#D55E00',mec = 'k',mew = 0.2)
for [idx,val] in enumerate([ax1, ax2]):
    val.set_xlabel("Actual (cm)")
    val.set_title(MLNames[idx] + ":\nWTD Predicted vs Actual (OOB)")
    val.annotate('$R^2$: ' + str(round(OOBr2[idx],2)) + "\nRMSE: " + str(round(OOBrmse[idx],2)), xy = (0.7,0.1),xycoords = 'axes fraction')
    
facecols = ['#0072B2','#D55E00','#0072B2','#D55E00']
# Loop to plot and annotate bar plots
for [idx,val] in enumerate([ax3, ax4, ax5, ax6]):
    val.set_xlim([-1, X.shape[1]])
    val.set_xticks(range(X.shape[1]), SortedNames[idx],rotation=270)
    val.bar(range(X.shape[1]), importances[idx], align="center",fc = facecols[idx],ec = 'k')
    val.set_title(importancetitles[idx] + " Feature Importances")
    
#Annotate the subplots
labellist1 = ["a).","b).","c).","d).","e).","f)."]
for [idx,val] in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
    val.annotate(labellist1[idx], xy = (0.925,0.925),xycoords = 'axes fraction',fontweight = 'bold')
# Save Figure
plt.savefig("Output_Figures/Predictions_Significance.pdf",dpi = 600) 

# Set up all models to calculae prediction intervals (using X prediction model)
all_models = {}
# Calculate Prediction Intervals    
for alpha in [0.05, 0.5, 0.95]:
    modelGB = GradientBoostingRegressor(loss="quantile",alpha = alpha,n_estimators = 500, max_depth = None, max_leaf_nodes = 12,
    random_state = rs, learning_rate = 0.05)
    all_models["q %1.2f" % alpha] = modelGB.fit(X, y)  

#%% Save GB model to import with make_predictions.py script
with open('GB_Model.pkl', 'wb') as file:
    pickle.dump(GB_Model, file)

with open('all_models.pkl', 'wb') as file:
    pickle.dump(all_models, file)
