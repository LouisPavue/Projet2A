# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:53:17 2020

@author: yassine.sameh
"""

import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time as t
from sklearn.preprocessing import StandardScaler
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



datas = pd.DataFrame()

    
def KNN(k):
    print("KNN")
    
    print("Chargement...")
    
    # lecture des donnees
    df = pd.read_csv('data/scalledValues_test.csv', header=0)
    
    
    """
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
    """
   
    
    label = {'decollage': 0,
             'atterrissage': 1,            
             'procedure': 2,
             'croisiere':3,
             'virage':4
             } 
    
    df.label = [label[item] for item in df.label] 
    
    lst = df["callsign"].unique()
    
    global datas
    
    Lat_index = 2
    Lon_index = 3
    Velocity_index = 4
    Heading_index = 5
    VertSpeed_index = 6
    Alt_index = 7
    
    if(datas.shape[0] == 0):
        for sign in tqdm(lst):
            temp = df[ df["callsign"].str.strip() == sign.strip()]
            
            #----------- Concatenation des valeurs pour chaques variables ---------------
            Values = list(temp.iloc[:,Lat_index].values)
            Values += list(temp.iloc[:,Lon_index].values)
            Values += list(temp.iloc[:,Velocity_index].values)
            Values += list(temp.iloc[:,Heading_index].values)
            Values += list(temp.iloc[:,VertSpeed_index].values)
            Values += list(temp.iloc[:,Alt_index].values)
            
            
            
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
    
    """
    X_test = X_test[0,:].reshape(1,-1)
    y_test = y_test[0:1]
    """
    clf = KNeighborsClassifier(n_neighbors=k)
    start_time = t.time()
    clf.fit(X_train , y_train)
    
    predictions = clf.predict(X_test)
    # evaluation du classifieur
    #cnf_matrix = confusion_matrix(predictions, y_test, labels=[0,1,2,3,4,5,6,7,8])
    cnf_matrix = confusion_matrix(predictions, y_test, labels=[0,1,2,3,4])
    
    """
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
    """
    
    
    index = ["decollage",
             "atterrissage", 
             'procedure',
             'croisiere',
             'virage'
            ]  
    columns =["decollage",
             "atterrissage", 
             'procedure',
             'croisiere',
             'virage'
            ]  
    
    cm_df = pd.DataFrame(cnf_matrix,columns,index)
    sns.heatmap(cm_df, annot=True,cmap="YlGnBu")
    print(cnf_matrix)
    print(classification_report(predictions , y_test))
    print('Accuracy: %.2f' % accuracy_score(y_test, predictions))
    display_runtime(start_time)# -*- coding: utf-8 -*-
    plt.show()
    score = []
    for i in range(1,15):
        clf = KNeighborsClassifier(n_neighbors=i)    
        clf.fit(X_train , y_train)       
        predictions = clf.predict(X_test)
        score.append(accuracy_score(y_test, predictions))
        
        
    plt.plot(score)
    plt.show()
        
