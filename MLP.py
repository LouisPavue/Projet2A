#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:59:38 2020

@author: yassine.sameh
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from math import *

import matplotlib.pyplot as plt

import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.preprocessing import QuantileTransformer

import time as t
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def display_runtime(start):
    run_time = t.time() - start
    hours = run_time // 3600
    minutes = (run_time % 3600) // 60
    seconds = (run_time % 3600) % 60
    result = ""
    if hours > 0:
        result = str(int(hours)) + "h "
    if minutes > 0:
        result += str(int(minutes)) + "min "
    result += str(int(seconds)) + "s "
    result += str(round((seconds-int(seconds))*1000)) + "ms"
    print('Elapsed Time : ', result)
    
def MLP():
    print("Multi Layer Perceptron")
    start_time = t.time()
    print("Chargement...")
    
    # lecture des donnees
    df = pd.read_csv('data/scalledValues.csv', header=0)
    
    
    lst = df["callsign"].unique()
    datas = pd.DataFrame()
    Lat_index = 2
    Lon_index = 3
    Velocity_index = 4
    Heading_index = 5
    VertSpeed_index = 6
    Alt_index = 7
    
    nVariable= 6
    
    for sign in tqdm(lst):
        temp = df[ df["callsign"].str.strip() == sign.strip()]
        
        #----------- Concatenation des valeurs pour chaques variables ---------------
        Values = list(temp.iloc[:,Lat_index].values)
        Values += list(temp.iloc[:,Lon_index].values)
        Values += list(temp.iloc[:,Velocity_index].values)
        Values += list(temp.iloc[:,Heading_index].values)
        Values += list(temp.iloc[:,VertSpeed_index].values)
        Values = list(temp.iloc[:,Alt_index].values)
        
        
        
        label = temp["label"].unique()[0]
        
        #-----------  ---------------
        Values.insert(0,sign.strip())
        
        Values += [label]
        
        speed_row = pd.Series(Values)
        
        speed_df = pd.DataFrame([speed_row])
        
        
        
        datas = pd.concat([datas, speed_df], ignore_index=True)
        

    # creation des ensembles train / test
    X_train, X_test, y_train, y_test = train_test_split(datas.iloc[:,1:-1],datas.iloc[:,-1], 
                                                        test_size=0.1, random_state=42)
  
    
    #Normalisationd des donnees
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    #Normalisation des données d'entrainement et de test
    StandardScaler(copy=True, with_mean=True, with_std=True)
       
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
 
    # creation du classifieur de reseau de neurones multicouches
    perceptron = MLPClassifier(
                               hidden_layer_sizes=(180*nVariable), 
                               max_iter=300,
                               activation = 'relu',
                               solver='adam',
                               random_state=1
                               )
    
    perceptron.fit(X_train , y_train)
    
    predictions = perceptron.predict(X_test)
    # evaluation du classifieur
    cnf_matrix = confusion_matrix(predictions, y_test, labels=["decollage", 
                                                               "atterrissage",
                                                               "virage",
                                                               "procedure",
                                                               "croisiere"])
    
    index = ['decollage','atterrissage','virage','procedure','croisiere']  
    columns = ['decollage','atterrissage','virage','procedure','croisiere']  
    cm_df = pd.DataFrame(cnf_matrix,columns,index)
    sns.heatmap(cm_df, annot=True,cmap="YlGnBu")
    print(cnf_matrix)
    print(classification_report(predictions , y_test))
    print('Accuracy: %.2f' % accuracy_score(y_test, predictions))
    display_runtime(start_time)