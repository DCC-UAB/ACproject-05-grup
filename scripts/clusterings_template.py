'''
En aquest fitxer executem i visualitzem l'algorisme de
clustering 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plots
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
# df_no_objectius = df_loaders.load_no_objectius()

# Carregar datasets guardats pickle
df = pd.read_pickle('pickles/df.pk1')
df_max_scaled = pd.read_pickle('pickles/df_max_scaled.pk1')
df_min_max_scaled = pd.read_pickle('pickles/df_min_max_scaled.pk1') 
df_final = pd.read_pickle('pickles/df_final.pk1')
df_no_objectius = pd.read_pickle('pickles/df_no_objectius.pk1')

# ===============================================================================================
...