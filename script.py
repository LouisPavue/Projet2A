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
labels = pd.read_csv("data/label2_test.csv", header =0, engine='python')
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
        extract = df[ df["callsign"].str.strip() == sign]
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
    
    f = open("data/scalledValues_test.csv","w")
    f.write("time,callsign,lat,lon,velocity,heading,vertrate,baroaltitude,label\n")
    for i in tqdm(range(0,len(X))):
        f.write(str(X.loc[i]['time'])+","+X.loc[i]['callsign']+","+str(X.loc[i]['lat'])+","+str(X.loc[i]['lon'])+","+str(X.loc[i]['velocity'])+","+str(X.loc[i]['heading'])+","+str(X.loc[i]['vertrate'])+","+str(X.loc[i]['baroaltitude'])+","+X.loc[i]['label']+"\n")
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

#createLabelisedCSV()
scalled = pd.read_csv("data/scalledValues_test.csv", header =0, engine='python')
"""
output = convertCSV(df[0:100])

print("géneration d'un fichier de test")
f = open("X_testGenerated.csv","w")
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
"""
"""


pt = PowerTransformer(method="azr")

#---------------- Visualisation 2D ---------------

#représentation graphique de tous les vols

lst = X[X["callsign"].str.strip() == "N319EP" ]
lst = lst["callsign"].unique()

for sign in tqdm(lst):
    print(sign)
    temp = X[ X["callsign"].str.strip() == "N319EP"]
    Xp = temp.iloc[: ,:].values
    
    plt.figure(figsize=(15, 3))
    plt.subplot(131)
    #vitesse en fonction du temps
    plt.scatter (Xp[:,0] , Xp[:,3] )
    plt.xlabel (df.columns[0])
    plt.ylabel (df.columns[3])
    
    plt.subplot(132)    
    #longitude en fonction de la latitude
    plt.scatter (Xp[:,1] , Xp[:,2] )
    plt.xlabel (df.columns[1])
    plt.ylabel (df.columns[2])
    plt.suptitle(sign)
    
    plt.subplot(133)    
    #longitude en fonction de la latitude
    plt.scatter (Xp[:,0] , Xp[:,7])
    plt.xlabel (df.columns[0])
    plt.ylabel (df.columns[7])
    plt.suptitle(sign)
    plt.show ()
    
 
lst = df[df["callsign"].str.strip() == "N319EP" ]
lst = lst["callsign"].unique()

for sign in tqdm(lst):
    print(sign)
    temp = df[ df["callsign"].str.strip() == "N319EP"]
    Xp = temp.iloc[: ,:].values
    
    plt.figure(figsize=(15, 3))
    plt.subplot(131)
    #vitesse en fonction du temps
    plt.scatter (Xp[:,0] , Xp[:,3] )
    plt.xlabel (df.columns[0])
    plt.ylabel (df.columns[3])
    
    plt.subplot(132)    
    #longitude en fonction de la latitude
    plt.scatter (Xp[:,1] , Xp[:,2] )
    plt.xlabel (df.columns[1])
    plt.ylabel (df.columns[2])
    plt.suptitle(sign)
    
    plt.subplot(133)    
    #longitude en fonction de la latitude
    plt.scatter (Xp[:,0] , Xp[:,7])
    plt.xlabel (df.columns[0])
    plt.ylabel (df.columns[7])
    plt.suptitle(sign)
    plt.show ()
        
    #ax = sns.pairplot(temp)




lst = df["callsign"].unique()
lst = lst[0:10]
verticalSpeed = pd.DataFrame()
verticalSpeedEnd = pd.DataFrame()
headings = pd.DataFrame()
positions = pd.DataFrame()

nTakeOff_colTest = 10
Vspeed_index = 6
Vheading = 5


print("Traitement des données")
f = open("dumb.csv", "w")
f.write("callsign , label1 , Mean vertate begin , label2 , Mean vertate end , label 3 , Heading, Heading inverse \n")
for sign in tqdm(lst):
  
    temp = df[ df["callsign"].str.strip() == sign.strip()]
    #vertical speed values for a callsign
    Vd = list(temp.iloc[0:nTakeOff_colTest,Vspeed_index].values)
    Vf = list(temp.iloc[-nTakeOff_colTest-1:-1,Vspeed_index].values)
    
    heading = list(temp.iloc[:,Vheading].values)
    heading_delta = (max(heading) - min(heading))   #sens horraire
    heading_delta_inverse = 360 - (max(heading) - min(heading))   #sens anti-horraire
    
    t_positions = (temp.iloc[:,2:4].values)
    for i in range(0, len(t_positions)):
        point = (t_positions[i,0],t_positions[i,1])
        flag = np.isin(point,t_positions[i:,:])
        #print(flag)
        
    
    #adding the callsign to the list of speed
    Vd.insert(0,sign.strip())
    Vf.insert(0,sign.strip())
    heading.insert(0,sign.strip())
    #convert to an array for de df
    speed_row = pd.Series(Vd)
    speed_row_end = pd.Series(Vf)
    heading_array = pd.Series(heading)
    
    if(speed_row.size - 1 < nTakeOff_colTest):
        var = speed_row[speed_row.size - 1]
        for i in range(speed_row.size, nTakeOff_colTest+1):
            speed_row[i] = var
     
        
    if(speed_row_end.size - 1 < nTakeOff_colTest):
        var = speed_row_end[speed_row.size - 1]
        for i in range(speed_row_end.size, nTakeOff_colTest+1):
            speed_row_end[i] = var   
            
    
            
    #speed df
    speed_df = pd.DataFrame([speed_row])
    speed_df_end = pd.DataFrame([speed_row_end])
    heading_df = pd.DataFrame([heading_array])
    
    
    #concatenation of both dataframes
    verticalSpeed = pd.concat([verticalSpeed, speed_df], ignore_index=True)
    verticalSpeedEnd = pd.concat([verticalSpeedEnd, speed_df_end], ignore_index=True)
    headings = pd.concat([headings, heading_df], ignore_index=True)
    
    
        
    
    #verticalSpeed.replace(None,speed_df[-1])
    #print(verticalSpeed)
    label = None
    somme = 0
    for c in range(0, nTakeOff_colTest):
        somme += float(speed_df[1+c])
        
    moyenne = somme / nTakeOff_colTest
    if(abs(moyenne) < 3):
        label = "palier"
    elif(moyenne >= 3):
        label = "montée"
    else:
        label = "descente"
    f.write(str(sign)+" , "+label+" , "+str(moyenne)+" , ")
    
    label = None
    somme = 0
    for c in range(0, nTakeOff_colTest):
        somme += float(speed_df_end[1+c])
        
    moyenne = somme / nTakeOff_colTest
    if(abs(moyenne) < 3):
        label = "palier"
    elif(moyenne >= 3):
        label = "montée"
    else:
        label = "descente"
    
    f.write(label+" , "+str(moyenne)+" , ")
    
    
    label = None
    if( (min(heading_delta,heading_delta_inverse) > 8 ) and (min(heading_delta,heading_delta_inverse) <= 90)):
        label = "virage simple"
    elif(min(heading_delta,heading_delta_inverse) <= 8 ):
        label = "rectiligine"
    else:
        label = "procedure"
        
    f.write(label+" , "+str(heading_delta)+" , "+ str(heading_delta_inverse)+" \n")
    #---------------- Visualisation 3D ---------------
    
    
print("END")
    
f.close()

test = pd.read_csv("dumb.csv", header =0, engine='python')

#--------------- Vitesse Verticale ----------------
"""
"""
fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
aze = df[ df["callsign"].str.strip() == ("KAL893")]
#zline = np.linspace(0, 15, 1000)
#xline = np.sin(zline)
#yline = np.cos(zline)
#ax.plot3D(xline, yline, zline, 'gray')

zdata = aze.iloc[:,13]
xdata = aze.iloc[:,2]
ydata = aze.iloc[:,3]
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

plt.show ()
"""