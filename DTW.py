#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:53:17 2020

@author: yassine.sameh
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from math import *
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.preprocessing import QuantileTransformer
import time as t
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance


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


"""
def dtw(s1, s2):
    DTW = np.zeros((len(s1)+1,len(s2)+1))
    DTW[:, 0] = np.inf
    DTW[0, :] = np.inf 
    DTW[0, 0] = 0

    for i in range(1, DTW.shape[0]):
        for j in range(1, DTW.shape[1]):
            dist= (s1[i-1]-s2[j-1])**2
            DTW[i, j] = dist + min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])
    return math.sqrt(DTW[len(s1), len(s2)]), DTW
"""

def dtw(a, b):   
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    return cumdist[an, bn]
    
def DTW_Classifier():
    print("DTW")
    start_time = t.time()
    print("Chargement...")
    
    # lecture des donnees
    df = pd.read_csv('data/scalledValues.csv', header=0)
    
    
    label = {'decollage': 0,
             'atterrissage': 1,
             'virage_montee': 2,
             'virage_descente': 3,
             'virage_croisiere': 4,
             'procedure': 5,
             'monte_croisiere': 6,
             'descente_croisiere': 7,
             'croisiere':8
             } 
  

    df.label = [label[item] for item in df.label] 
    
    lst = df["callsign"].unique()
    datas = pd.DataFrame()
    Lat_index = 2
    Lon_index = 3
    Velocity_index = 4
    Heading_index = 5
    VertSpeed_index = 6
    Alt_index = 7
    
    
    for sign in tqdm(lst):
        temp = df[ df["callsign"].str.strip() == sign.strip()]
        
        #----------- Concatenation des valeurs pour chaques variables ---------------
        Values = list(temp.iloc[:,Lat_index].values)
        Values += list(temp.iloc[:,Lon_index].values)
        Values += list(temp.iloc[:,Velocity_index].values)
        Values += list(temp.iloc[:,Heading_index].values)
        Values += list(temp.iloc[:,VertSpeed_index].values)
        Values = list(temp.iloc[:,Alt_index].values)
        
        
        
        label_string = temp["label"].unique()[0]
        
        #-----------  ---------------
        Values.insert(0,sign.strip())
        
        Values += [label_string]
        
        speed_row = pd.Series(Values)
        
        speed_df = pd.DataFrame([speed_row])
        
        
        
        datas = pd.concat([datas, speed_df], ignore_index=True)
        

    # creation des ensembles train / test
    X_train, X_test, y_train, y_test = train_test_split(datas.iloc[:,1:-1],datas.iloc[:,-1], 
                                                        test_size=0.2, random_state=42)
  
    
    #Normalisationd des donnees
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    #Normalisation des données d'entrainement et de test
    StandardScaler(copy=True, with_mean=True, with_std=True)
       
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    parameters = {'n_neighbors':[1]}
    clf = GridSearchCV(KNeighborsClassifier(metric=dtw), parameters, cv=3, verbose=True)
    
    clf.fit(X_train , y_train)
    
    predictions = clf.predict(X_test)
    # evaluation du classifieur
    cnf_matrix = confusion_matrix(predictions, y_test, labels=[0,1,2,3,4,5,6,7,8])
    
    index = ["decollage","atterrissage",
             "virage_montee",
             "virage_descente",
             "virage_croisiere",
             'procedure',
             'monte_crosiere',
             'descente_croisiere',
             'croisiere'
            ]  
    columns =["decollage","atterrissage",
             "virage_montee",
             "virage_descente",
             "virage_croisiere",
             'procedure',
             'monte_crosiere',
             'descente_croisiere',
             'croisiere'
            ] 
   
    cm_df = pd.DataFrame(cnf_matrix,columns,index)
    sns.heatmap(cm_df, annot=True,cmap="YlGnBu")
    print(cnf_matrix)
    print(classification_report(predictions , y_test))
    print('Accuracy: %.2f' % accuracy_score(y_test, predictions))
    display_runtime(start_time)# -*- coding: utf-8 -*-
"""
mat = [[4 ,0 ,3 ,0 ,0 ,0 ,0 ,0 ,0],
     [0 ,4 ,0 ,4 ,0 ,3, 0, 0, 0],
     [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],
     [0 ,3 ,0 ,4 ,0 ,1 ,1 ,1 ,1],
     [0 ,0 ,1 ,1 ,0 ,0 ,2 ,0 ,1],
     [0 ,3 ,0 ,1 ,0 ,2 ,0 ,0 ,0],
     [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],
     [2 ,3 ,0 ,5 ,0 ,0 ,0 ,1 ,0],
     [0 ,0 ,0 ,0 ,3 ,0 ,0 ,2 ,8]]
"""