'''
Aquest script serà importat en altres scripts. La seva funció
és fer plots de diferentes característiques en moments determinats.
'''

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

def plot_heatmap(df_original, df_normalized, clusters, variables, k_def):
    df_norm_copy = df_normalized.copy()
    df_orig_copy = df_original.copy()
    
    for k, cluster_labels, _ in clusters:
        if k == k_def:
            df_norm_copy['Cluster'] = cluster_labels
            df_orig_copy['Cluster'] = cluster_labels
            break

    mean_values_real = df_orig_copy.groupby('Cluster')[variables].mean()
    mean_values_norm = df_norm_copy.groupby('Cluster')[variables].mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(mean_values_norm, annot=mean_values_real, cmap='coolwarm', linewidths=0.5, fmt='.2f')

    plt.title(f'Heatmap of Mean Values by Cluster (k={k_def})')
    plt.xlabel('Variables')
    plt.ylabel('Clusters')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def plot_sorted_classified_clusters(df_original, clusters, health_variables, k_def, threshold=None):
    df_copy = df_original.copy()
    for k, cluster_labels, _ in clusters:
        if k == k_def:
            df_copy['Cluster'] = cluster_labels
            break
    
    mean_health = df_copy.groupby('Cluster')[health_variables].mean()
    mean_health['health_score'] = mean_health.sum(axis=1)
    
    if threshold is None:
        threshold = mean_health['health_score'].mean()  # Mitjana general com a llindar
    
    mean_health['health_status'] = mean_health['health_score'].apply(lambda x: 'Mal estat' if x > threshold else 'Bon estat')
    sorted_clusters = mean_health['health_score'].sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_clusters.index, y=sorted_clusters.values, palette="coolwarm")
    plt.axhline(threshold, color='red', linestyle='--', label=f"Llindar (threshold = {threshold:.2f})")
    plt.title("Clústers ordenats per estat de salut mental (pitjor a millor)")
    plt.xlabel("Clúster")
    plt.ylabel("Puntuació de salut mental (sumatori)")
    plt.xticks(rotation=0)
    plt.legend()
    plt.show()

    print(mean_health[['health_score', 'health_status']])


def plot_tsne_clusters(df, clusterss, k_def):
    print(f"Carregant plot tsne 3D amb k = {k_def}")
    df_t = df.copy()
    for k, clusters, _ in clusterss:
            if k == k_def:
                tsne = TSNE(n_components=3, random_state=42, n_iter=1000)
                tsne_result = tsne.fit_transform(df_t)
                tsne_df = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2', 'Dim3'])
                tsne_df['Cluster'] = clusters

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                for cluster in range(k_def):
                    cluster_data = tsne_df[tsne_df['Cluster'] == cluster]
                    ax.scatter(cluster_data['Dim1'], cluster_data['Dim2'], cluster_data['Dim3'], label=f'Cluster {cluster}')
            
                ax.set_title('Clusters visualitzats amb t-SNE (3D)')
                ax.set_xlabel('Dim1')
                ax.set_ylabel('Dim2')
                ax.set_zlabel('Dim3')
                ax.legend()
                plt.show()

def plot_tsne_clusters_2D(df, clusterss, k_def):
    print(f"Carregant plot tsne 2D amb k = {k_def}")
    df_t = df.copy()
    for k, clusters, _ in clusterss:
        if k == k_def:
            # Canviem n_components a 2 per obtenir t-SNE en 2D
            tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
            tsne_result = tsne.fit_transform(df_t)
            
            # Crear DataFrame amb els resultats de t-SNE
            tsne_df = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2'])
            tsne_df['Cluster'] = clusters

            # Crear el gràfic en 2D
            fig, ax = plt.subplots(figsize=(10, 8))

            # Visualitzar cada cluster
            for cluster in range(k_def):
                cluster_data = tsne_df[tsne_df['Cluster'] == cluster]
                ax.scatter(cluster_data['Dim1'], cluster_data['Dim2'], label=f'Cluster {cluster}')
            
            # Ajustar etiquetes i títol
            ax.set_title('Clusters Visualitzats amb t-SNE (2D)')
            ax.set_xlabel('Dim1')
            ax.set_ylabel('Dim2')
            ax.legend()
            plt.show()


def plot_pca_clusters(df, clusterss, k_def):
    df_t = df.copy()

    for k, clusters, _ in clusterss:
        if k == k_def:
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(df_t)
            pca_df = pd.DataFrame(pca_result, columns=['Dim1', 'Dim2', 'Dim3'])
            pca_df['Cluster'] = clusters

            # 3D Plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            for cluster in range(k_def):
                cluster_data = pca_df[pca_df['Cluster'] == cluster]
                ax.scatter(cluster_data['Dim1'], cluster_data['Dim2'], cluster_data['Dim3'], label=f'Cluster {cluster}')
            
            ax.set_title('Clusters Visualized with PCA (3D)')
            ax.set_xlabel('Dim1')
            ax.set_ylabel('Dim2')
            ax.set_zlabel('Dim3')
            ax.legend()
            plt.show()


def plot_pca_clusters_2D(df, clusterss, k_def):
    df_t = df.copy()

    for k, clusters, _ in clusterss:
        if k == k_def:
            # Aplicar PCA per a la reducció de dimensionalitat a 2D
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_t)
            
            # Crear DataFrame amb els resultats de PCA
            pca_df = pd.DataFrame(pca_result, columns=['Dim1', 'Dim2'])
            pca_df['Cluster'] = clusters

            # Gràfic en 2D
            fig, ax = plt.subplots(figsize=(10, 8))

            # Visualitzar cada cluster
            for cluster in range(k_def):
                cluster_data = pca_df[pca_df['Cluster'] == cluster]
                ax.scatter(cluster_data['Dim1'], cluster_data['Dim2'], label=f'Cluster {cluster}')
            
            # Ajustar títol i etiquetes
            ax.set_title('Clusters Visualitzats amb PCA (2D)')
            ax.set_xlabel('Dim1')
            ax.set_ylabel('Dim2')
            ax.legend()
            plt.show()