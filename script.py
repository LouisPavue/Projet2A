#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
# lecture des donnees
df = pd.read_csv("data/01.csv", header =0)
# nettoyage des donnees
df = df.dropna()
df.index = np.arange(len(df))

  

print("END")
        