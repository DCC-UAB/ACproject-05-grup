'''
Aquest fitxer permet executar i visualitzar totes les funcionalitats principals del 
directòri.
'''
import os
import sys
import pandas as pd
from scripts import df_loaders
from scripts.xgboost_ import calcular_importancia_xgboost

def clear():
    if os.name == 'nt':  #  Windows
        os.system('cls')
    else:  # Linux & macOS
        os.system('clear')



print('''
===========================================================      
======  Anàlisi i clusterització de la salut mental  ======
======  dels estudiants mitjançant Machine Learning  ======
===========================================================    
      ''')

if os.path.exists('pickles/df.pk1'):
    print("S'han trobat dataframes pickle, es carregaràn automàticament...")
    df = pd.read_pickle('pickles/df.pk1')
    df_max_scaled = pd.read_pickle('pickles/df_max_scaled.pk1')
    df_min_max_scaled = pd.read_pickle('pickles/df_min_max_scaled.pk1')
    df_final = pd.read_pickle('pickles/df_final.pk1')
    df_no_objectius = pd.read_pickle('pickles/df_no_objectius.pk1')
else:
    print("No s'han trobat fitxers pickles per carregar els dataframes. Es carregaràn de nou.")
    df = df_loaders.load_df()
    df_max_scaled = df_loaders.load_max_scaled()
    df_min_max_scaled = df_loaders.load_min_max_scaled()
    df_final = df_loaders.load_final()
    df_no_objectius = df_loaders.load_no_objectius()
print("Dataframes carregats")


match int(input('''
Escull una opció:
    1. Feature Importance
    2. Clustering
               -> ''')):
    case 1:
        match int(input('''
[Has escollit: Feature Importance]
    Escull quin algorisme vols fer servir:
        1. Random Forest
        2. XGBoost
                -> ''')):
            case 1: # Random Forest
                ...
            
            case 2: # XGBoost
                ...

    case 2:
        match int(input('''
[Has escollit: Clustering]
    Escull quin algorisme vols fer servir:
        1. Kmeans
        2. Gmm (Gaussian Mixture Model)
        3. Agglomerative Clustering
                -> ''')):
            case 1: # Kmeans
                ...
            case 2: # Gmm
                ...
            case 3: # Agglomerative
                ...



