#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:53:41 2024 @author: eoin.reddin@universityofgalway.ie
For detailed description refer to README.md
"""
from osgeo import gdal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def Geotiff_2_RF_Array(ADates, DDates, Filters, cols2drop, Days2AVG, FilterEnv,
                       WTDThresholds, pathlist, Band_Names):
    #%% SubFunctions
    # Function to read in the multi-spectral tifs
    def readColocatedData(FileName,PZlocations):
        dataset = gdal.Open(FileName)
        # Get raster dimensions
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        bands = dataset.RasterCount
        # Initialize empty array
        MSarray = np.zeros((bands, height, width))
        # Accessing individual bands and assign to array
        for i in range(1, bands+1):
            band = dataset.GetRasterBand(i)
            MSarray[i-1,:,:] = band.ReadAsArray()
        # Get geotransform information
        geotransform = dataset.GetGeoTransform()
        # Format Longitude and Latitude arrays
        Lon = geotransform[0] + np.arange(0, width) * geotransform[1]
        Lat = geotransform[3] + np.arange(0, height) * geotransform[5]
        Latidx = np.zeros(len(PZlocations.ycoord))
        Lonidx = np.zeros(len(PZlocations.xcoord))
        # Find locations of each piezometer
        for [idx,val] in enumerate(PZlocations.ycoord):
            Latidx[idx] = (np.abs(Lat - val)).argmin()
            Lonidx[idx] = (np.abs(Lon - PZlocations.xcoord[idx])).argmin()
        return(MSarray,Lat,Lon,Latidx,Lonidx,bands)
    # Format Met Data
    def formatmet(METdf,inputCSV,t,idx):
        DateTemp = inputCSV[inputCSV.date.isin(t)]
        METdf.loc[idx, 'date'] = val
        METdf.loc[idx, 'maxtp'] = pd.to_numeric(DateTemp.maxtp).mean()
        METdf.loc[idx, 'soil'] = pd.to_numeric(DateTemp.soil).mean()
        METdf.loc[idx, 'rain'] = pd.to_numeric(DateTemp.rain).mean()
        return(METdf)
    
    #%% Data Set up    
    # List of files namesto import
    TifFileNames = [pathlist[0] + val + ".tif" for val in ADates] + [pathlist[1] +val + ".tif" for val in DDates]

    # Convert to datetime
    Dates = [datetime.strptime(val, "%d-%m-%Y") for val in ADates] + [datetime.strptime(val, "%d-%m-%Y") for val in DDates] 
    #%% Open Piezometers
    # Paths to piezometer locations, and all water table data
    PZlocations = pd.read_csv(pathlist[2])
    WTD = pd.read_csv(pathlist[3])
    WTD.datetime = pd.to_datetime(WTD.datetime,format = "%d/%m/%Y %H:%M")
    # Initialise arrays for indexing later
    PZlocations["Average_WTD"] = 1.0

    if Filters[0] == "Y":
        PZlocations = PZlocations[~PZlocations.Well_ID.str.contains("_D")].reset_index()

    PZlocs2, MSarrayout, Latidxout, Lonidxout = ([] for i in range(4))

    #%% Read Met data
    AthenryCSV = pd.read_csv(pathlist[4],skiprows=24)
    DillonCSV = pd.read_csv(pathlist[5],skiprows=24)
    for [idx,val] in enumerate([AthenryCSV,DillonCSV]):
        val.date = pd.to_datetime(val.date,format = "%d-%b-%Y")

    #%% Format Piezometers (nested arrays as we are working with multiple multiband images)
    METMtDillon = pd.DataFrame(columns = ['date','maxtp','soil','rain'])
    METAthenry = pd.DataFrame(columns = ['date','maxtp','soil','rain'])

    for [idx,val] in enumerate(Dates):
        for [jdx,jval] in enumerate(PZlocations.Well_ID):
            # Find all values on certain day
            WTD_Dates = WTD[(WTD.datetime >= val)  & (WTD.datetime <= val + timedelta(hours = 23))]
            # Find WTD at each piezometer on each day and calculate mean
            WTD_Dates =  WTD_Dates[WTD_Dates['WellName'].str.contains(jval)]
            PZlocations.loc[jdx,"Average_WTD"] = WTD_Dates.loc[:, 'Average of corrected_level'].mean()
        # Append means to have list for each date
        PZlocs2.append(PZlocations.dropna(ignore_index = True))
        #Amount of time before each date to average over
        t = np.arange(val - timedelta(days = Days2AVG), val + timedelta(days = 1), timedelta(days=1)).astype(datetime)
        # Set up met data
        METMtDillon = formatmet(METMtDillon,DillonCSV,t,idx)
        METAthenry = formatmet(METAthenry,AthenryCSV,t,idx)

    #%% Open the GeoTIFF files
    for [idx,val] in enumerate(TifFileNames):
        [MSarray,Lat,Lon,Latidx,Lonidx,bands] = readColocatedData(val,PZlocs2[idx])
        #"*out" means it contains all dates
        MSarrayout.append(MSarray)
        Latidxout.append(Latidx.astype(int))    
        Lonidxout.append(Lonidx.astype(int))    
    #%% Extract data from each location
    # Initialise data frame
    DF_4_RF = pd.DataFrame(columns = Band_Names)  
    DF_4_RF.dropna(axis=1, how='all')
    list2append = [0]*19
    # Loop to create dataframe containing multispectral parameter at each piezometer, and WTD
    for [jdx,val] in enumerate(MSarrayout):
        DF2append = pd.DataFrame(np.zeros((len(Latidxout[jdx]),len(Band_Names))),columns = Band_Names)  # Initialise array to append
        for [idx,val] in enumerate(Latidxout[jdx]):
            list2append[0:15] = MSarrayout[jdx][:,val,Lonidxout[jdx][idx]] # Multispec values        
            list2append[15] = PZlocs2[jdx].Average_WTD[idx] # Add relevent WTD to the end    
            list2append[16] = PZlocs2[jdx].Dist2Bund[idx] # Add distance to Bund to the end
            list2append[17] = PZlocs2[jdx].Dist2Dam[idx] # Add distance to Cell the end
            list2append[18] = PZlocs2[jdx].Peat_Depth[idx] # Add peatdepth the end  
            DF2append.loc[idx] = list2append #append each list to dataframe for each date
        # Add met data to random forest 
        if Filters[1] == "Y":
            DF2append = DF2append.join(pd.DataFrame({'Soil-MTDill.': [METMtDillon.soil[jdx]] * len(Latidxout[jdx])}))
            DF2append = DF2append.join(pd.DataFrame({'MaxT-Ath.': [METAthenry.maxtp[jdx]] * len(Latidxout[jdx])}))      
        DF_4_RF = pd.concat([DF_4_RF.dropna(axis=1, how='all'), DF2append.dropna(axis=1, how='all')],ignore_index=True) # append dataframe to Random Forest dataframe (.dropna included to supress warning)

    #%% Filter out cloudy indices    
    DF_4_RF = DF_4_RF[DF_4_RF.CloudMask != 1] # Values not equal to one (1s are clouds)
    DF_4_RF = DF_4_RF.drop('CloudMask', axis=1)
        
    #%% Apply Filters
    if FilterEnv == "Y":
        DF_4_RF = DF_4_RF[DF_4_RF.AVG_WTD >= WTDThresholds[0]].dropna(how='all').reset_index(drop=True)
        DF_4_RF = DF_4_RF[DF_4_RF.AVG_WTD <= WTDThresholds[1]].dropna(how='all').reset_index(drop=True)

    if Filters[0] == "Y":
       DF_4_RF.drop(cols2drop, inplace=True, axis=1) 

    return(DF_4_RF)