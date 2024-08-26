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

def Geotiff_2_Predict(Dates, Filters, cols2drop, Days2AVG, FilterEnv,
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
    def formatmet(METdf,inputCSV,t,idx):
        DateTemp = inputCSV[inputCSV.date.isin(t)]
        METdf.loc[idx, 'date'] = val
        METdf.loc[idx, 'maxtp'] = pd.to_numeric(DateTemp.maxtp).mean()
        METdf.loc[idx, 'soil'] = pd.to_numeric(DateTemp.soil).mean()
        METdf.loc[idx, 'rain'] = pd.to_numeric(DateTemp.rain).mean()
        return(METdf)
    
    #%% Data Set up  
    # List of files namesto import
    TifFileNames = [pathlist[0] +val + ".tif" for val in Dates]

    # Convert to datetime
    Dates = [datetime.strptime(val, "%d-%m-%Y") for val in Dates] 
    #%% Open Piezometers
    # Initialise arrays for indexing later
    PZlocations = pd.read_csv(pathlist[1])

    PZlocs2, MSarrayout, Latidxout, Lonidxout = ([] for i in range(4))

    #%% Read Met data
    AthenryCSV = pd.read_csv(pathlist[3],skiprows=24)
    DillonCSV = pd.read_csv(pathlist[4],skiprows=24)
    for [idx,val] in enumerate([AthenryCSV,DillonCSV]):
        val.date = pd.to_datetime(val.date,format = "%d-%b-%Y")

    #%% Format Piezometers (nested arrays as we are working with multiple multiband images)
    METMtDillon = pd.DataFrame(columns = ['date','maxtp','soil','rain'])
    METAthenry = pd.DataFrame(columns = ['date','maxtp','soil','rain'])

    for [idx,val] in enumerate(Dates):
        # #Amount of time before each date to average over
        t = np.arange(val - timedelta(days = Days2AVG), val + timedelta(days = 1), timedelta(days=1)).astype(datetime)
        # Set up met data
        METMtDillon = formatmet(METMtDillon,DillonCSV,t,idx)
        METAthenry = formatmet(METAthenry,AthenryCSV,t,idx)

    #%% Open the GeoTIFF files
    [MSarrayout,Lat,Lon,Latidxout,Lonidxout,bands] = readColocatedData(TifFileNames[0],PZlocations)
    Latidxout = Latidxout.astype(int)  
    Lonidxout = Lonidxout.astype(int) 
    #%% Extract data from each location
    # Initialise data frame
    DF_4_RF = pd.DataFrame(columns = Band_Names)  
    DF_4_RF.dropna(axis=1, how='all')
    list2append = [0]*18
    DF2append = pd.DataFrame(np.zeros((len(Latidxout),len(Band_Names))),columns = Band_Names)  # Initialise array to append
    for [idx,val] in enumerate(Latidxout):
        list2append[0:15] = MSarrayout[:,val,Lonidxout[idx]] # Multispec values 
        list2append[15]= PZlocations.Dist2Bund[idx] # Add distance to Bund to the end
        list2append[16] = PZlocations.Dist2Dam[idx] # Add distance to Cell the end
        list2append[17] = PZlocations.Peat_Depth[idx] # Add peatdepth the end
        DF2append.loc[idx] = list2append #append each list to dataframe for each date
    # Add met data to random forest 
    if Filters[1] == "Y":
        DF2append = DF2append.join(pd.DataFrame({'Soil-MTDill.': [METMtDillon.soil[0]] * len(Latidxout)}))
        DF2append = DF2append.join(pd.DataFrame({'MaxT-Ath.': [METAthenry.maxtp[0]] * len(Latidxout)}))
    DF_4_RF = DF2append.dropna(axis=1, how='all')

    #%% Filter out cloudy indices  
    PZlocations = PZlocations[DF_4_RF.CloudMask != 1] # Values not equal to one (1s are clouds)
    DF_4_RF = DF_4_RF[DF_4_RF.CloudMask != 1] # Values not equal to one (1s are clouds)
    DF_4_RF = DF_4_RF.drop('CloudMask', axis=1)
        
    #%% Apply Filters
    if Filters[0] == "Y":
       DF_4_RF.drop(cols2drop, inplace=True, axis=1) 

    return(PZlocations.ycoord,PZlocations.xcoord,DF_4_RF)