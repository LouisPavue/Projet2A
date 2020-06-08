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
    

#--------------- Vitesse Verticale ----------------
#%%
nMin= 42
nMax = nMin + 10

lst = df["callsign"].unique()[nMin:nMax]
#df = df[ df["callsign"].str.strip() == "UAE223"]
#lst = df["callsign"].unique()

for sign in tqdm(lst):
    print("\n"+sign)
    temp = df[ df["callsign"].str.strip() == sign.strip()]
    
    Xp = temp.iloc[: ,:].values

    plt.figure(figsize=(10, 3))
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
    print("Alt :\n")
    print(Xp[:,7])
