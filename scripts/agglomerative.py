'''
En aquest fitxer executem i visualitzem l'algorisme de
clustering Agglomerative
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
df_file = 'pickles/df.pk1'
df = pd.read_pickle(df_file)

df_max_scaled_file = 'pickles/df_max_scaled.pk1'
df_max_scaled = pd.read_pickle(df_max_scaled_file)

df_min_max_scaled_file = 'pickles/df_min_max_scaled.pk1'
df_min_max_scaled = pd.read_pickle(df_min_max_scaled_file)

df_final_file = 'pickles/df_final.pk1'
df_final = pd.read_pickle(df_final_file)

df_no_objectius_file = 'pickles/df_no_objectius.pk1'
df_no_objectius = pd.read_pickle(df_no_objectius_file)

# ===============================================================================================
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def agglomerative_clustering(dfs, max_k=10):
    df_t = dfs.copy()

    sil_scores = []
    clusterings = []

    sil_best_score = -1
    sil_best_k = 2

    for k in range(2, max_k + 1):
        # Crear el model d'Agglomerative Clustering
        model = AgglomerativeClustering(n_clusters=k)
        clusters = model.fit_predict(df_t)
        
        # Silhouette score
        if k > 2:
            score = silhouette_score(df_t, clusters)
            sil_scores.append(score)
            if score > sil_best_score:
                sil_best_score = score
                sil_best_k = k
        else:
            sil_scores.append(None)  # per k = 2 no es fa el calcul

        clusterings.append((k, clusters, 0))

    # Mostrar el millor k segons silhouette score
    print("Millor k silhouette:", sil_best_k)

    # Gràfic del Silhouette score per cada k
    plt.figure(figsize=(8, 6))
    k_values = range(2, max_k + 1)
    plt.plot(k_values, sil_scores, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score per a diferents k')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()
    
    return clusterings

c = agglomerative_clustering(df_no_objectius, 8)

# Escollim millor k -> 3

k_def = 3 # Canviar per veure els reusltats d'un clustering amb una k diferent

plots.plot_tsne_clusters(df_no_objectius, c, k_def)

plots.plot_heatmap(df, df_min_max_scaled, c, ['sex', 'cesd', 'stai_t', 'mbi_ex', 'part', 'year', 'job', 'health', 'psyt', 'mbi_ea'], k_def)