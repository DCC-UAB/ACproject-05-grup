'''
En aquest fitxer executem i visualitzem l'algorisme de
clustering gmm
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
from sklearn.mixture import GaussianMixture

def gmm(dfs, max_k=10):
    df_t = dfs.copy()

    bic_scores = []
    clusterings = []

    best_bic = float('inf')
    best_k = 2

    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm.fit(df_t)
        bic = gmm.bic(df_t)
        bic_scores.append(bic)

        clusters = gmm.predict(df_t)

        if bic < best_bic:
            best_bic = bic
            best_k = k
        
        cluster_centers = gmm.means_
        clusterings.append((k, clusters, cluster_centers))
    
    print(f"Best k according to BIC: {best_k}")

    plt.figure(figsize=(8, 6))
    k_values = range(2, max_k + 1)
    plt.plot(k_values, bic_scores, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('BIC Score')
    plt.title('BIC Scores for Gaussian Mixture Models')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    return clusterings

if __name__ == '__main__':
    import plots
    import df_loaders

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

    c = gmm(df_no_objectius, 8)

    # Escollim millor k -> 5 
    
    k_def = 5 # Canviar per veure els reusltats d'un clustering amb una k diferent

    plots.plot_tsne_clusters(df_no_objectius, c, k_def)

    # plots.plot_tsne_clusters_2D(df_no_objectius, c, k_def)

    plots.plot_heatmap(df, df_max_scaled, c, ['sex', 'cesd', 'stai_t', 'mbi_ex', 'part', 'year', 'job', 'health', 'psyt', 'mbi_ea'], k_def)

    ###

