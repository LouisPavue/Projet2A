#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import time, datetime
# lecture des donnees
df = pd.read_csv("data/01.csv", header =0)
# nettoyage des donnees
df = df.dropna()    #suppression des cases nulles
df = df[ df["callsign"].str.strip() != ("")]    #suppression des cases avec callsign == "   "

df.index = np.arange(len(df))   #réagencement des index

#for i in tqdm(range(0, len(df))):
    #df.iloc[:,0] = datetime.datetime.fromtimestamp(df.iloc[:,0]).utcnow()

# https://germain-forestier.info/teaching/files/FD4/09-hierarchique.pdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist , squareform
from scipy.cluster.hierarchy import linkage , dendrogram

print("END")
        

