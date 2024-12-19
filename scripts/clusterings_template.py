'''
En aquest fitxer executem i visualitzem l'algorisme de
clustering 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import df_loaders

import math
import time


'''
IMPORTANT
 En cas que no funcioni, cal assegurar-se que estem executant el fitxer des de 
 ACPROJECT-05-GRUP
 Per comprovar des d'on executem podem fer servir os.getcwd()
 Si en el output es veu que estem executant des de dins d'una carpeta:
    - "c:\\Users\\joanc\\OneDrive\\Desktop\\ACproject-05-grup\\dataset1"
 Podem arreglar-ho descomentant la segÜnet línia del script. Serveix tant per 
 Windows com per Linux.
 Per qualsevol dubte, se'ns pot contactar des dels correus específicats en 
 el README
 '''
# os.chdir('..)


# # Carregar dataset des de csv
# df = df_loaders.load_df()
# df_max_scaled = df_loaders.load_max_scaled()
# df_min_max_scaled = df_loaders.load_min_max_scaled()
# df_final = df_loaders.load_final()

# Carregar datasets guardats pickle
df_file = "pickles/dfs/df.pk1"
df = pd.read_pickle(df_file)

df_max_scaled_file = "pickles/dfs/df_max_scaled.pk1"
df_max_scaled = pd.read_pickle(df_max_scaled_file)

df_min_max_scaled_file = "pickles/dfs/df_min_max_scaled.pk1"
df_min_max_scaled = pd.read_pickle(df_min_max_scaled_file)

df_final_file= "pickles/dfs/df_final.pk1"
df_final = pd.read_pickle(df_final_file)

# ===============================================================================================
...