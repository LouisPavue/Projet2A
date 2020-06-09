#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:07:31 2020

@author: louispavue
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import accuracy_score

import time as t


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
    
def scheduler(epoch, lr):
   if epoch < 10:
     return lr
   else:
     return lr * keras.math.exp(-0.1)

def CNN():
    df = pd.read_csv('data/scalledValues_test.csv', header=0)
    lst = df["callsign"].unique()
    
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
    
    
    Lat_index = 2
    Lon_index = 3
    Velocity_index = 4
    Heading_index = 5
    VertSpeed_index = 6
    Alt_index = 7
    
    var = [Lat_index,Lon_index,Velocity_index,Heading_index,VertSpeed_index,Alt_index]
    y = len(lst)
    x = 181
    z = len(var)
    
    #cube = np.zeros((y,z,x))
    som = 0
    n = 1
    
    
    cube = np.zeros((z,y,x))
    j = 0
    for sign in tqdm(lst):
        temp = df[ df["callsign"].str.strip() == sign.strip()]
        
        for k in range(0,z): #valeur a l'instant x variable k  
            variable = var[k]          
            Values = list(temp.iloc[:,variable].values)
            for i in range(0,x-1): #temp
                cube[k,j,i] = Values[i]
            cube[k,j,i+1] = temp["label"].unique()[0]
        j+=1
        
    sc =  MinMaxScaler(feature_range = (-1,1))
    
    cube_scaled = cube
    for i in range(0,z):
        cube_scaled[i,:,0:180] = sc.fit_transform(cube_scaled[i,:,0:180])    
        
           
    cube = cube.swapaxes(0,1)
    cube = cube.swapaxes(2,1)
    
    
    #x_train, x_test, y_train, y_test = train_test_split(cube[:,:,0:-1],cube[:,:,-1:], 
    #                                                        test_size=0.2, random_state=42)
    
    
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=50)
    train_index, test_index = list(split.split(cube[:,0:-1,0],cube[:,-1,0]))[0]
    #len(train_index), len(test_index)    
    
    x_train, y_train = cube[train_index,0:-1 ,:], cube[train_index,-1:,0]
    x_test, y_test = cube[test_index,0:-1 ,:], cube[test_index,-1:,0]
    """
    x_train =  cube[0:91,0:-1,:]
    y_train =  cube[0:91:,-1:,0]
    #y_train = y_train.reshape()
    
    x_test = cube[91:,0:-1,:]
    y_test = cube[91:,-1:,0]
    """
    
    
    
    # transform the labels from integers to one hot vectors
    """
    enc = OneHotEncoder()
    enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))
    y_train = enc.transform(y_train.reshape(-1,1)).toarray()
    y_test = enc.transform(y_test.reshape(-1,1)).toarray()
    
    train = np.ndarray(shape=(z,91,9), dtype=float, order='F')
    test = np.ndarray(shape=(z,36,9), dtype=float, order='F')
    
    """
    train = np.ndarray(shape=(len(y_train),6), dtype=float, order='F')
    test = np.ndarray(shape=(len(y_test),6), dtype=float, order='F')
    
    for i in range(0,len(train)):
        for j in range(0,len(var)):
            train[i,j] = y_train[i,0]
       
    for i in range(0,len(test)):
        for j in range(0,len(var)):
            test[i,j] = y_test[i,0]   
    #test[:,:] = y_test[:,:]
        
    y_train = train
    y_test = test
    
    input_shape = x_train.shape[1:]
    
    n_feature_maps = 64
    
    input_layer = keras.layers.Input(input_shape)
    
    # BLOCK 1
    
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    
    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    
    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    
    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    
    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)
    
    # BLOCK 2
    
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    
    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    
    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    
    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    
    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)
    
    # BLOCK 3
    
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    
    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    
    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    
    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)
    
    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)
    
    # FINAL
    
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    
    output_layer = keras.layers.Dense(z, activation='softmax')(gap_layer)
    
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    model = keras.Sequential([model])
    
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)  #0.0001
    
    file_path = 'best_model.hdf5'
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                        save_best_only=True)
    
    #callbacks = [reduce_lr, model_checkpoint]
    callbacks = keras.callbacks.LearningRateScheduler(scheduler)
    
    # We fixed the number of epochs to 2000: meaning there will be 2000 
    # training passes over the whole dataset    1000
    nb_epochs = 1200
    
    # the batch size is fixed also: the dataset is divided to 12 batches 
    # for each batch we will apply a gradient descent and update the parameters 
    mini_batch_size = 12
    """
    f = open("crossvamidation.csv","w")
    for i in range(0,100,10):
        for j in range(0,100,10):
    """
            # we call the fit function and provide the corresponding hyperparameters and callbacks
    start_time = t.time()
    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                          verbose=True,validation_data=(x_test,y_test) , callbacks=[reduce_lr,model_checkpoint])
    
    #, callbacks=[reduce_lr,model_checkpoint]
    model = keras.models.load_model('best_model.hdf5')
    
    loss, acc = model.evaluate(x_test, y_test)
    
    print('Test accuracy', acc)
    
    
    metric = 'loss'
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_'+metric])
    plt.title('model '+metric)
    plt.ylabel(metric,fontsize='large')
    plt.xlabel('epoch',fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.close()
    
    
    
    #predictions = model.predict_classes(x_test)
    
    y_classes = model.predict_classes(x_test)
    
    print('Accuracy: %.2f' % accuracy_score(y_test[:,0], y_classes))
    
    #cnf_matrix = confusion_matrix(y_classes, y_test[:,0], labels=[0,1,2,3,4,5,6,7,8])
    cnf_matrix = confusion_matrix(y_classes, y_test[:,0], labels=[0,1,2,3,4])
    
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
    plt.show()
    #print(cnf_matrix)
    #print(classification_report(y_classes , y_test[:,0]))
    print('Accuracy: %.2f' % accuracy_score(y_test[:,0], y_classes))
    display_runtime(start_time)
    
    #print("Mean accuracy : "+str(som/n))
    
    
    
