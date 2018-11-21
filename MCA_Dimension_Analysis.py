#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:35:21 2018

@author: slowking
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import prince


Pokedex_Master = pd.read_csv("pokedex.csv")
Combat_Master = pd.read_csv("combats_test.csv")
Stats_Master = pd.read_csv("pokemon.csv")
Attacker = pd.read_csv("chart.csv")
Evolution = pd.read_csv("pokemon_species.csv")

Master_NoStats_Col = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,24,29,30,31,39]
is_Legendary = Pokedex_Master.drop(Pokedex_Master.columns[Master_NoStats_Col], axis=1)


is_Legendary['capture_rate'] = pd.to_numeric(is_Legendary['capture_rate'], errors='coerce')

is_Legendary[['type1','type2']] = is_Legendary[['type1','type2']].fillna(value='None')
is_Legendary['capture_rate'] = pd.to_numeric(is_Legendary['capture_rate'], errors='coerce')

is_Legendary.info()
    #Convert Types with One-Hot Encoding

Pokedex_Types=is_Legendary[['pokedex_number','type1','type2']]

labelencoder = LabelEncoder()
Pokedex_Types['T1']= labelencoder.fit_transform(Pokedex_Types.type1)
Pokedex_Types['T2']= labelencoder.fit_transform(Pokedex_Types.type2)

t1_ohe = OneHotEncoder()
t2_ohe = OneHotEncoder()
t1_ohe_array = t1_ohe.fit_transform(Pokedex_Types.T1.values.reshape(-1,1)).toarray()
t2_ohe_array = t2_ohe.fit_transform(Pokedex_Types.T2.values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(t1_ohe_array, columns = ["type1_"+str(int(i)) for i in range(t1_ohe_array.shape[1])])
Pokedex_Types = pd.concat([Pokedex_Types, dfOneHot], axis=1)

dfOneHot = pd.DataFrame(t2_ohe_array, columns = ["type2_"+str(int(i)) for i in range(t2_ohe_array.shape[1])])
Pokedex_Types = pd.concat([Pokedex_Types, dfOneHot], axis=1)

Pokedex_Types_PCA = Pokedex_Types.drop(Pokedex_Types.columns[[0,1,2,3,4]], axis=1)

mca = prince.MCA(n_components=37, n_iter=100, copy=False, engine='auto', random_state=42)
mca = mca.fit(Pokedex_Types_PCA)
print(np.sum(mca.explained_inertia_))

Types_MCA = is_Legendary[['type1','type2']]
mca2 = mca.fit(Types_MCA)
print(np.sum(mca2.explained_inertia_))


pca2 = prince.PCA(n_components=9, engine='sklearn', rescale_with_mean=False, rescale_with_std=False,)
pca3 = pca2.fit(Pokedex_Types_PCA)
print(np.sum(pca3.explained_inertia_))





