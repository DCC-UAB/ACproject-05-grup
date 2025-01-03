'''
En aquest fitxer executem i visualitzem l'algorisme de
clustering mini-batch kmeans
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import math
import time


'''
IMPORTANT
 En cas que no funcioni, cal assegurar-se que estem executant el fitxer des de 
 ACPROJECT-05-GRUP
 Per comprovar des d'on executem podem fer servir os.getcwd()
 Si en el output es veu que estem executant des de dins d'una carpeta:
    - "c:\\Users\\joanc\\OneDrive\\Desktop\\ACproject-05-grup\\dataset1"
 Podem arreglar-ho descomentant la següent línia del script. Serveix tant per 
 Windows com per Linux.
 Per qualsevol dubte, se'ns pot contactar des dels correus específicats en 
 el README
 '''
# os.chdir('..)    

# ===============================================================================================
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

def mini_batch_kmeans(dfs, max_k=10):
    df_t = dfs.copy()

    inertia = []
    clusterings = []

    sil_best_score = -1
    sil_best_k = 2
    size_batch = int(dfs.shape[0]/2)
    beg = time.time()
    for k in range(2, max_k + 1):
        model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=size_batch, max_iter=100)
        clusters = model.fit_predict(df_t)
        
        # Silhouette
        if k > 2:
            score = silhouette_score(df_t, clusters)
            if score > sil_best_score:
                sil_best_score = score
                sil_best_k = k
        
        inertia.append(model.inertia_)
        cluster_centers = model.cluster_centers_
        clusterings.append((k, clusters, cluster_centers))
    print(f"Acabat en {time.time() - beg} seogns")
    # Silhouette 
    print("Best k silhouette:", sil_best_k)

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

if __name__ == '__main__':
    import plots
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

    c = mini_batch_kmeans(df_no_objectius, 8)

    # Escollim millor k -> 3

    k_def = 3 # Canviar per veure els resultats d'un clustering amb una k diferent

    plots.plot_tsne_clusters(df_no_objectius, c, k_def)

    plots.plot_heatmap(df, df_min_max_scaled, c, ['cesd', 'stai_t', 'mbi_ex', 'mbi_cy', 'psyt', 'part', 'year', 'job', 'health', 'qcae_cog', 'stud_h', 'mbi_ea'], k_def)

    plots.plot_sorted_classified_clusters(df, c, ['cesd', 'stai_t', 'mbi_ex'], k_def)