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

#df = df[ df["callsign"].str.strip() == ("THY6233")]
#df = df[ df["squawk"] == (7000)]
df.index = np.arange(len(df))   #réagencement des index

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
    X = temp.iloc[: ,0:5].values
    
    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    #vitesse en fonction du temps
    plt.scatter (X[:,0] , X[:,4] )
    plt.xlabel (df.columns[0])
    plt.ylabel (df.columns[4])
    plt.subplot(122)
    
    #longitude en fonction de la latitude
    plt.scatter (X[:,2] , X[:,3] )
    plt.xlabel (df.columns[2])
    plt.ylabel (df.columns[3])
    plt.suptitle(sign)
    #plt.show ()
    
    #ax = sns.pairplot(temp)

"""


lst = df["callsign"].unique()
lst = lst[0:10]
verticalSpeed = pd.DataFrame()
nTakeOff_colTest = 10
Vspeed_index = 6


print("Traitement des données")
f = open("decollage.csv", "w")
for sign in tqdm(lst):
  
    temp = df[ df["callsign"].str.strip() == sign.strip()]
    #vertical speed values for a callsign
    V = list(temp.iloc[0:nTakeOff_colTest,Vspeed_index].values)
    #adding the callsign to the list of speed
    V.insert(0,sign.strip())
    #convert to an array for de df
    speed_row = pd.Series(V)
    #speed df
    speed_df = pd.DataFrame([speed_row])
    #concatenation of both dataframes
    verticalSpeed = pd.concat([verticalSpeed, speed_df], ignore_index=True)
    #print(verticalSpeed)
    label = None
    somme = 0
    for c in range(0, nTakeOff_colTest):
        somme += speed_df[1+c]
        
    moyenne = somme / nTakeOff_colTest
    if(abs(moyenne[0]) < 3):
        label = "palier"
    elif(moyenne[0] >= 3):
        label = "montée"
    else:
        label = "descente"
    f.write(str(sign)+" , "+label+" , "+str(moyenne[0])+"\n")
    
    #---------------- Visualisation 3D ---------------
    
    
print("END")
    
f.close()

test = pd.read_csv("decollage.csv", header =0, engine='python')

#--------------- Vitesse Verticale ----------------
"""

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
#zline = np.linspace(0, 15, 1000)
#xline = np.sin(zline)
#yline = np.cos(zline)
#ax.plot3D(xline, yline, zline, 'gray')

zdata = df.iloc[:,13]
xdata = df.iloc[:,2]
ydata = df.iloc[:,3]
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Reds');

plt.show ()
"""