'''
En aquest fitxer executem i visualitzem l'algorisme de
clustering Kmeans
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
df_file = "pickles/dfs/df.pk1"
df = pd.read_pickle(df_file)

df_max_scaled_file = "pickles/dfs/df_max_scaled.pk1"
df_max_scaled = pd.read_pickle(df_max_scaled_file)

df_min_max_scaled_file = "pickles/dfs/df_min_max_scaled.pk1"
df_min_max_scaled = pd.read_pickle(df_min_max_scaled_file)

df_final_file = "pickles/dfs/df_final.pk1"
df_final = pd.read_pickle(df_final_file)

df_no_objectius_file = "pickles/dfs/df_no_objectius.pk1"
df_no_objectius = pd.read_pickle(df_no_objectius_file)

# ===============================================================================================
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans(dfs, max_k=10):
    df_t = dfs.copy()

    inertia = []
    clusterings = []

    sil_best_score = -1
    sil_best_k = 2

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = model.fit_predict(df_t)
        
        # Silhouette
        if k > 2:
            score = silhouette_score(df_t, clusters)
            if score > sil_best_score:
                sil_best_score = score
                sil_best_k = k
         ##

        inertia.append(model.inertia_)
        cluster_centers = model.cluster_centers_
        clusterings.append((k, clusters, cluster_centers))

    # Silhouette 
    print("Millor k silhouette:", sil_best_k)

    plt.figure(figsize=(8, 6))
    k_values = range(2, max_k + 1)
    plt.plot(k_values, inertia, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()
    
    return clusterings

c = kmeans(df_no_objectius, 8)

# Escollim millor k -> 3

k_def = 3

plots.plot_tsne_clusters(df_no_objectius, c, k_def)

plots.plot_heatmap(df, df_min_max_scaled, c, ['sex', 'cesd', 'stai_t', 'mbi_ex', 'part', 'year', 'job', 'health', 'psyt', 'mbi_ea'], k_def)