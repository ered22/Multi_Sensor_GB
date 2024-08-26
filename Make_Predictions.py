#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:14:40 2024 @author: eoin.reddin@universityofgalway.ie
For detailed description refer to README.md
"""
# Import modules
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec
from Geotiff_2_Predict import Geotiff_2_Predict
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

# Make scatter plots of predicted values
def prediction_plots(bogname,predicted_date, Filters, cols2drop, Days2AVG, FilterEnv, WTDthresholds, pathlist, all_models):

    [axlist1,axlist2] = formatsubplots()    # Function to set up subplot grid
    # Format Castlegar Data
    pathlist = prep_txt_files(bogname,"Predicting_Data","Zero_Day_D","PZ_Predicted")
    del pathlist[1]
    # Labels for annotation    
    labels = {"Castlegar": (["a).","b).","c).","d)."], ["e).","f).","g).","h)."]), "Clooneeny": (["i).","j).","k).","l)."], ["m).","n).","o).","p)."])}
    labellist1, labellist2 = labels.get(bogname, ([], []))
    # Loop to make predictions      
    for [idx,val] in enumerate(predicted_date):
        # Band names for each layer in GEOTIFF **N.B these must match those exported from SNAP**
        Band_Names = ["VV","NDVI","NDWI","STR","SAVI","CloudMask","TCG","TCW","Blue","Green","Red","NIR","SWIR1","SWIR2","VH","Dist2Bund","Dist2Dam","PeatDepth"]
        # Prepare data for Prediction
        [Pred_Lat_Cas,Pred_Lon_Cas,DF_4_Pred_Cas] = Geotiff_2_Predict([val], Filters, cols2drop, Days2AVG, FilterEnv, WTDthresholds, pathlist, Band_Names)
        NewPredsCas = GB_Model.predict(DF_4_Pred_Cas)
        
        print('\n'+ bogname+' ' + val +' WTD [Min, Med, Mean, Max]: [' + str(round(min(NewPredsCas),2))+ ", " + str(round(np.median(NewPredsCas),2))  
              + ", " + str(round(np.mean(NewPredsCas),2)) + ", " + str(round(max(NewPredsCas),2)) + "]\n")
        
        # Great Than vals
        Sphagnum_Water = NewPredsCas >= -20
        fraction_trues = np.sum(Sphagnum_Water) / (np.sum(Sphagnum_Water) + len(Sphagnum_Water) - np.sum(Sphagnum_Water))
        
        # Make Subplot of figures for multiple dates
        plot1 = axlist1[idx].scatter(Pred_Lon_Cas,Pred_Lat_Cas, 0.5,NewPredsCas,marker='.',vmin = -60,vmax = 0,cmap = 'YlGnBu')    
        
        # Create a colormap with two colors, define the bounds of the color categories and normalize
        cmap = mcolors.ListedColormap(['#FFFFCC', '#1C4814'])
        bounds = [0, 1, 2]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Make Figure
        plot2 = axlist2[idx].scatter(Pred_Lon_Cas,Pred_Lat_Cas, 0.5,Sphagnum_Water.astype(int),marker='.',norm=norm,cmap = cmap)
        if idx ==3:
            # Create a colorbar with the correct bounds
            cbar = plt.colorbar(plot2, ticks=[0, 0.5, 1.5])
            cbar.ax.set_yticklabels(['','< 0.2 m', '> 0.2 m'],rotation = 270,verticalalignment = 'center')  
            
        # Label both sets of subplots
        axlist1[idx].set_title(val + '\n' + bogname + " WTD"), axlist2[idx].set_title("Shallow WTD" )
        # Call formatting function
        predicted_scatter(axlist1[idx],axlist2[idx],idx,labellist1[idx],labellist2[idx],plot1,NewPredsCas,fraction_trues,bogname)
        axlist2[idx].annotate("WTD > 0.2 m: \n" + str(round(np.mean(fraction_trues)*100,2))+"%", xy = (0.05,0.05),xycoords = 'axes fraction')
    # Label Y axis    
    axlist1[0].set_ylabel('Latitude'),  axlist2[0].set_ylabel('Latitude')
    plt.savefig("Output_Figures/"+ bogname +"Predictions.png",dpi = 300)

    [axlist3,axlist4] = formatsubplots()    # Function to set up subplot grid
    # Make Prediction Intervals
    for [idx,val] in enumerate(predicted_date):
        # Prepare data for Prediction
        [Pred_Lat_Cas,Pred_Lon_Cas,DF_4_Pred_Cas] = Geotiff_2_Predict([val], Filters, cols2drop, Days2AVG, FilterEnv, WTDthresholds, pathlist, Band_Names)
        NewPredsCas = GB_Model.predict(DF_4_Pred_Cas)
        y_lower = all_models["q 0.05"].predict(DF_4_Pred_Cas)
        y_upper = all_models["q 0.95"].predict(DF_4_Pred_Cas)
        # Make Subplot of figures for multiple dates
        plot1 = axlist3[idx].scatter(Pred_Lon_Cas,Pred_Lat_Cas, 0.5,y_lower/100,marker='.',vmin = -0.60,vmax = 0,cmap = 'YlGnBu')    
        # Make Figure
        axlist4[idx].scatter(Pred_Lon_Cas,Pred_Lat_Cas, 0.5,y_upper/100,marker='.',vmin = -0.60,vmax = 0,cmap = 'YlGnBu')
        # Label both sets of subplots
        axlist3[idx].set_title(val + '\n' +"Lower Pred. Interval"), axlist4[idx].set_title("Upper Pred. Interval" )
        # Call formatting function
        predicted_scatter(axlist3[idx],axlist4[idx],idx,labellist1[idx],labellist2[idx],plot1,y_lower,y_upper,bogname)
        # Annotate Means
        axlist4[idx].annotate("Mean (m): \n" + str(round(np.mean(y_upper/100),2)), xy = (0.05,0.05),xycoords = 'axes fraction')
    # Label Y axis    
    axlist3[0].set_ylabel('Latitude'),  axlist4[0].set_ylabel('Latitude')  
    plt.savefig("Output_Figures/"+ bogname +"Prediction_Intervals.png",dpi = 300)
    return

# Used to remove duplication in the prediction_plots function
def predicted_scatter(axes1,axes2,idx,label1,label2,plot1,mean1,mean2,bogname):
        axes1.grid(color='gainsboro', linestyle='-', linewidth=0.25), axes2.grid(color='gainsboro', linestyle='-', linewidth=0.25)
        # Annotate Means
        axes1.annotate("Mean (m): \n" + str(round(np.mean(mean1/100),2)), xy = (0.05,0.05),xycoords = 'axes fraction')
        # Annotate Plots
        axes1.annotate(label1, xy = (0.875,0.925),xycoords = 'axes fraction',fontweight = 'bold')
        axes2.annotate(label2, xy = (0.875,0.925),xycoords = 'axes fraction',fontweight = 'bold')
        axes1.set_xticklabels([])
        # Conditional formatting
        if idx != 0:
            axes1.set_yticklabels([])
            axes2.set_yticklabels([])
        if idx == 3:
            cbar = plt.colorbar(plot1)
            cbar.set_label('Predicted Water Table Depth (m)',rotation = 270,labelpad = 15)   
            # Set axis limits to resize if indices have been dropped by cloud masking 
        if bogname == "Clooneeny":
            axes1.set_ylim([53.691644452, 53.721585308]), axes1.set_xlim([-7.868829609750001, -7.837505355249999])  
            axes2.set_ylim([53.691644452, 53.721585308]), axes2.set_xlim([-7.868829609750001, -7.837505355249999])  
               
        axes2.set_xlabel('Longitude')
        return

# Function to set up tile of subplots
def formatsubplots():
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    gs = GridSpec(2, 4, figure=fig)
    ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),  fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])
    axlist1 = [ax7, ax8, ax9, ax10] 
    axlist2 = [ax11, ax12, ax13, ax14]
    return(axlist1,axlist2)
#%% Set up
# Make Predictions
predicted_date = ["25-04-2021","29-06-2021","21-11-2021","16-11-2022"]
predicted_date_cloon = ["10-09-2022","16-11-2022","22-06-2023","10-01-2024"]#,"21-11-2021"]
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

# Name of Bogs
bogname = ["Castlegar","Clooneeny"]

# Import Trained model, and prediction bounds model
with open('GB_Model.pkl', 'rb') as file1, open('all_models.pkl', 'rb') as file2:
    GB_Model = pickle.load(file1)
    all_models = pickle.load(file2)

# Path list
pathlist = prep_txt_files(bogname[1],"Zero_Day_A","Zero_Day_D","PZ")

#%% Make New Predictions and figures  
# Catlegar
prediction_plots(bogname[0],predicted_date, Filters, cols2drop, Days2AVG, FilterEnv, WTDthresholds, pathlist, all_models)
# Clooneeny
prediction_plots(bogname[1],predicted_date_cloon, Filters, cols2drop, Days2AVG, FilterEnv, WTDthresholds, pathlist, all_models)