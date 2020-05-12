#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import time, datetime
from math import *
# lecture des donnees
print("Lecture données ")
df = pd.read_csv("data/states_2019-12-23-00.csv", header =0, engine='python')
# nettoyage des donnees
df = df.dropna()    #suppression des cases nulles
df = df[ df["callsign"].str.strip() != ("")]    #suppression des cases avec callsign == "   "

#df = df[ df["callsign"].str.strip() == ("VOI941")]
#df = df[ df["squawk"] == (7000)]
df.index = np.arange(len(df))   #réagencement des index

aze = df[ df["callsign"].str.strip() == ("CTM0021")]
#for i in tqdm(range(0, len(df))):
    #df.iloc[:,0] = datetime.datetime.fromtimestamp(df.iloc[:,0]).utcnow()

# https://germain-forestier.info/teaching/files/FD4/09-hierarchique.pdf

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist , squareform
from scipy.cluster.hierarchy import linkage , dendrogram
import seaborn as sns
from mpl_toolkits import mplot3d




#---------------- Visualisation 2D ---------------
"""
#représentation graphique de tous les vols
lst = df["callsign"].unique()


for sign in tqdm(lst):
    temp = df[ df["callsign"].str.strip() == sign.strip()]
    X = temp.iloc[: ,0:6].values
    
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    #vitesse en fonction du temps
    plt.scatter (X[:,0] , X[:,4] )
    plt.xlabel (df.columns[0])
    plt.ylabel (df.columns[4])
    plt.subplot(132)
    
    #longitude en fonction de la latitude
    plt.scatter (X[:,2] , X[:,3] )
    plt.xlabel (df.columns[2])
    plt.ylabel (df.columns[3])
    plt.suptitle(sign)
    

    plt.show ()
    
    plt.scatter (X[:,0] , X[:,5] )
    plt.xlabel (df.columns[0])
    plt.ylabel (df.columns[5])
    plt.show()
    #ax = sns.pairplot(temp)

"""


lst = df["callsign"].unique()
lst = lst[0:10]
verticalSpeed = pd.DataFrame()
verticalSpeedEnd = pd.DataFrame()
headings = pd.DataFrame()

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

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
aze = df[ df["callsign"].str.strip() == ("JBU2136")]
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