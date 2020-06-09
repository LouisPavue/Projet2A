import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.metrics import Precision, Recall, AUC
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
import tensorflow.keras.backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



nVariable= 6

def mean_pred(y_true,y_pred):
    return K.mean(y_pred)

def build_fn(optimizer):
    model = Sequential()
    model.add(
        Dense(220*nVariable, input_dim=180*nVariable, activation="relu")
    )
    model.add(
        Dense(180*nVariable, input_dim=180*nVariable, activation="relu")
    )
    model.add(Dense(9, activation="softmax"))
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=[
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )
    return model


datas = pd.DataFrame()

def MLP_keras():
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
    global datas
    lst = df["callsign"].unique()
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
                                                        test_size=0.2, random_state=50)
  
    
    #Normalisationd des donnees
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    #Normalisation des données d'entrainement et de test
    StandardScaler(copy=True, with_mean=True, with_std=True)
       
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    
    input_dim = X_train.shape[1]
    nb_classes = 9
    """
    model = Sequential()
    model.add(Dense(nVariable*input_dim, input_dim=input_dim))
    model.add(Activation('sigmoid'))
    #model.add(Dropout(0.1))
    model.add(Dense(nVariable*input_dim))
    model.add(Activation('sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(nb_classes))


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',mean_pred])

    model.fit(X_train, y_train)

    predictions = model.predict_classes(X_test)
    """

    clf = KerasClassifier(build_fn, optimizer="rmsprop", epochs=120, batch_size=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    
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

