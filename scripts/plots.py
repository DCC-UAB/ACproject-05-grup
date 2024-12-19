'''
Aquest script serà importat en altres scripts. La seva funció
és fer plots de diferentes característiques en moments determinats.
'''

import seaborn as sns
import matplotlib.pyplot as plt

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