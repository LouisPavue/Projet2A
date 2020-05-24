#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from math import *

import matplotlib.pyplot as plt

import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.preprocessing import QuantileTransformer

# lecture des donnees
print("Lecture données ")

def parser(x):
	return datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S')

df = pd.read_csv("data/states_2019-12-23-00.csv", header =0, engine='python')
labels = pd.read_csv("data/label2.csv", header =0, engine='python')
# nettoyage des donnees
df = df.dropna()    #suppression des cases nulles
df = df[ df["callsign"].str.strip() != ("")]    #suppression des cases avec callsign == "   "

df = df.drop(columns=["icao24" , "onground" , "alert" , "spi" , "squawk" , "geoaltitude", "lastposupdate" , "lastcontact"])
    
#df = df[ df["callsign"].str.strip() == ("VOI941")]
nbpoints = 180
def createLabelisedCSV():
    print("Création d'un dataframe labelisé")
    X =  pd.DataFrame()
    
    maxim = 0
    #for i in tqdm(range(0,len(labels[labels["Callsign"].str.strip() == "ART9771"]))): 
    for i in tqdm(range(0,len(labels["Callsign"]))): 
        sign = labels["Callsign"][i]
        label = labels["Label"][i]
        extract = df[ df["callsign"].str.strip() == sign.strip()]
        extract.insert(len(df.columns),"label",label)
        #print(sign + " len : "+str(len(extract)))
        maxim = max(maxim,len(extract))
    
        extract['time'] = extract['time'].apply(lambda x: datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
        #extract.index  = extract['time']
        #extract = extract.resample('10S').bfill()[0:180]
        
     
        tmp = extract.iloc[-1]
        #last_value = tmp.values[-1]
        i = len(extract)
        nbpoints = 180
        if( i < nbpoints):
            goal_row = nbpoints
            while i < goal_row :
                extract = extract.append(tmp , ignore_index=False)
                i+=1
        elif( i >= nbpoints):
            extract_tmp = extract
            goal_row = nbpoints*2
            while i < goal_row :
                extract_tmp = extract_tmp.append(tmp , ignore_index=False)
                i+=1
            extract = pd.DataFrame()
            z = pd.DataFrame()
            z2 = pd.DataFrame()
            
            for j in range(0,goal_row,2):
                
                z = extract_tmp.iloc[j]
                z2 = extract_tmp.iloc[j+1]
                z = pd.DataFrame([z])
                z2 = pd.DataFrame([z2])
                z = pd.concat([z,z2])
                z = pd.DataFrame([z.mean()])
                z.insert(0,"time",[extract_tmp.iloc[j][0]])
                z.insert(6,"callsign",[extract_tmp.iloc[j][6]])
                z.insert(8,"label",[extract_tmp.iloc[j][8]])
                
                X = pd.concat([X,z])
    
        X = pd.concat([X, extract])
    
    X.index = np.arange(len(X))   #réagencement des index
    #print("Taille : "+str(maxim)) 
    
    f = open("data/scalledValues.csv","w")
    f.write("time,callsign,lat,lon,velocity,heading,vertrate,baroaltitude,label\n")
    for i in tqdm(range(0,len(X))):
        f.write(str(X.loc[i]['time'])
        +","+
        X.loc[i]['callsign']
        +","+str(X.loc[i]['lat'])
        +","+str(X.loc[i]['lon'])
        +","+str(X.loc[i]['velocity'])
        +","+str(X.loc[i]['heading'])
        +","+str(X.loc[i]['vertrate'])
        +","+str(X.loc[i]['baroaltitude'])
        +","
        +X.loc[i]['label']
        +"\n")
    f.close()
    
    print("end")    

    
def convertCSV(inputDF):
    print("convertion du DataFrame")
    lst = inputDF["callsign"].unique()
    #lst = df["callsign"].unique()[0:10]
   
    Lat_index = 1
    Lon_index = 2
    Velocity_index = 3
    Heading_index = 4
    VertSpeed_index = 5
    Alt_index = 7
    out_df = pd.DataFrame()
    X = pd.DataFrame()
    for sign in tqdm(lst):
       # print(sign)
 
        temp = df[ df["callsign"].str.strip() == sign.strip()]
        temp['time'] = temp['time'].apply(lambda x: datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
        tmp = temp.iloc[-1]
        i = len(temp)
        
        if( i < nbpoints):
            goal_row = nbpoints
            while i < goal_row :
                temp = temp.append(tmp , ignore_index=False)
                i+=1
        elif( i >= nbpoints):
            extract_tmp = temp
            goal_row = nbpoints*2
            while i < goal_row :
                extract_tmp = extract_tmp.append(tmp , ignore_index=False)
                i+=1
            temp = pd.DataFrame()
            z = pd.DataFrame()
            z2 = pd.DataFrame()
            
            for j in range(0,goal_row,2):
                
                z = extract_tmp.iloc[j]
                z2 = extract_tmp.iloc[j+1]
                z = pd.DataFrame([z])
                z2 = pd.DataFrame([z2])
                z = pd.concat([z,z2])
                z = pd.DataFrame([z.mean()])
                z.insert(0,"time",[extract_tmp.iloc[j][0]])
                z.insert(6,"callsign",[extract_tmp.iloc[j][6]])

                
                X = pd.concat([X,z])
    
        X = pd.concat([X, temp])
        
        temp = X[ X["callsign"].str.strip() == sign.strip()]
        #print(temp)
        Values = []
        Values = list(temp.iloc[:,Lat_index].values)
        Values += list(temp.iloc[:,Lon_index].values)
        Values += list(temp.iloc[:,Velocity_index].values)
        Values += list(temp.iloc[:,Heading_index].values)
        Values += list(temp.iloc[:,VertSpeed_index].values)
        Values += list(temp.iloc[:,Alt_index].values)
        Values.insert(0,sign.strip())
    #vertical speed values for a callsign
     
        row = pd.Series(Values)
        rowdf = pd.DataFrame([row])
        out_df = pd.concat([out_df, rowdf], ignore_index=True)
    print("END")
    return out_df

createLabelisedCSV()
scalled = pd.read_csv("scalledValues.csv", header =0, engine='python')

output = convertCSV(df[0:100])

print("géneration d'un fichier de test")
f = open("data/X_testGenerated.csv","w")
f.write("callsign,")
for i in range(1,len(output.columns)-1):
    f.write(str(i)+",")
f.write(str(len(output.columns)-1)+"\n")    

for i in tqdm(range(0, len(output))):
    for j in range(0, len(output.columns)-1):
        f.write(str(output.loc[i][j])+",")
    f.write(str(output.loc[i][j+1])+"\n")
f.close()

xtest = pd.read_csv("X_testGenerated.csv", header =0, engine='python')