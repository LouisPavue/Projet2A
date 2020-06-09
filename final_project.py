#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:28:21 2020

@author: yassine.sameh
"""

#==================== Import des modules necessaires==============
import sys
import time as t
from os import path


import script as datatreat
import KNN as knn
import MLP as mlp
import SVM as svm
import CNN_keras as cnn
import TREE as DecisionTree

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
    
    
# Main definition - constants
#menu_actions  = {}  

# =======================
#     MENUS FUNCTIONS
# =======================

# Main menu
def main_menu():
    
    print("===========================================")
    print("Projet Pattern Recognition Aviation Civile")
    print("===========================================\n")
    print("Choisir une méthode de classification :\n")
    print("1. Plus proches voisins")
    print("2. Support Vector Machine")
    print("3. Arbre de décision")
    print("4. Multi Layer Perceptron")
    print("5. Convolutional Neural Network")
    print("\n0. Quitter")
    choice = input(" >>  ")
    exec_menu(choice)
    return

# Back to main menu
def back():
    menu_actions['main_menu']()



#=============== Plus Proches Voisins ====================

def knn_function():
    knn.KNN()
    menu_actions['main_menu']()


#=============== Arbre de décision ====================

def DecisionTree_function():
    DecisionTree.TREE()
    menu_actions['main_menu']()


#================ SVM ============================================
def SVM_function():
    svm.SVM()
    menu_actions['main_menu']()


#=============== MultiLayer Perceptron ===========================
def MLP_function():
    mlp.MLP()
    menu_actions['main_menu']()
    
#=============== Convolutionnal Neural Network ===================
def CNN_function():
    cnn.CNN()
    menu_actions['main_menu']()


#======================== Menu ===============================================
# Executer le menu
def exec_menu(choice):
    
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Selection invalide, choisir une selection valide .\n")
            menu_actions['main_menu']()
    return


# Exit le programme
def exit():
    sys.exit()

# =======================
#    MENUS DEFINITIONS
# =======================

# Menu definition
menu_actions = {
    'main_menu': main_menu,
    '1': knn_function,
    '2': SVM_function,
    '3': DecisionTree_function,
    '4': MLP_function,
    '5': CNN_function,
    '9': back,
    '0': exit,
}


    
# =======================
#      MAIN PROGRAM
# =======================

# Main Program
if __name__ == "__main__":
    # Launch main menu
    main_menu()
    exist_data_scalled = path.exists("data/scalledValues.csv")
    if (not exist_data_scalled):
        datatreat.createLabelisedCSV
