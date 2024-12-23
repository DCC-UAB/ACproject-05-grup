import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from mpl_toolkits.mplot3d import Axes3D

# ======================================
# 1. Càrrega de datasets guardats (pickle)
# ======================================
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

# ======================================
# 2. Funció: Importància de les variables amb XGBoost
# ======================================
def calcular_importancia_xgboost(X, y, title="Importància de Variables"):
    model = XGBClassifier(random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    features = X.columns

    # Visualització
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color='skyblue')
    plt.title(title)
    plt.xlabel("Importància")
    plt.ylabel("Variables")
    plt.gca().invert_yaxis()
    plt.show()
    return pd.Series(importances, index=features)

# ======================================
# 3. Funció: Densitat de variables psicològiques per clúster
# ======================================
def calcular_densitat_clusters(df, clusters, variables):
    df['Cluster'] = clusters
    densitat = df.groupby('Cluster')[variables].mean()
    print("Densitat de variables psicològiques per clúster:\n", densitat)

    # Visualització heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(densitat, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Densitat de variables psicològiques per clúster")
    plt.show()

# ======================================
# 4. Funció: Clustering amb t-SNE
# ======================================
def tsne_clustering(X, n_clusters=3, method="KMeans"):
    tsne = TSNE(n_components=3, random_state=42)
    tsne_result = tsne.fit_transform(X)
    tsne_df = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2', 'Dim3'])
    
    # KMeans o GMM
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        model = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    
    tsne_df['Cluster'] = model.fit_predict(tsne_result)
    
    # Visualització
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cluster in range(n_clusters):
        cluster_data = tsne_df[tsne_df['Cluster'] == cluster]
        ax.scatter(cluster_data['Dim1'], cluster_data['Dim2'], cluster_data['Dim3'], label=f'Cluster {cluster}')
    
    ax.set_title(f'Clusters amb t-SNE (3D) - {method}')
    ax.set_xlabel('Dim1')
    ax.set_ylabel('Dim2')
    ax.set_zlabel('Dim3')
    ax.legend()
    plt.show()
    return tsne_df

# ======================================
# 5. Funció: Entrenar Regressió Logística
# ======================================
def entrenar_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # ROC i mètriques
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Visualització ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


# ======================================
# 6. Execució principal
# ======================================
if __name__ == "__main__":
    # Variables
    variables_psicologiques = ['cesd', 'stai_t', 'mbi_ex', 'mbi_cy']
    variables_academiques = ['year','health']
    variables_binàries = ['part', 'job', 'psyt']
    variables_numeriques = ['stud_h','age', 'jspe', 'qcae_cog', 'qcae_aff', 'erec_mean', 'mbi_ea']

    # Crear índex de salut mental
    df['mental_health_index'] = df[variables_psicologiques].mean(axis=1)
    llindar = df['mental_health_index'].median()
    df['mental_health_status'] = df['mental_health_index'].apply(lambda x: 1 if x > llindar else 0)

    # Preprocessament
    scaler = StandardScaler()
    X_no_psicologiques = pd.DataFrame(scaler.fit_transform(df[variables_academiques + variables_numeriques + variables_binàries]),
                                      columns=variables_academiques + variables_numeriques + variables_binàries)
    X_psicologiques = pd.DataFrame(scaler.fit_transform(df[variables_psicologiques]), columns=variables_psicologiques)
    X = pd.DataFrame(scaler.fit_transform(df[variables_academiques + variables_numeriques + variables_binàries]),
                     columns=variables_academiques + variables_numeriques + variables_binàries)
    y = df['mental_health_status']

    # 5.1 Importància de les variables
    print("=== Importància de Variables Psicològiques ===")
    calcular_importancia_xgboost(X_psicologiques, y, title="Importància de Variables Psicològiques")

    print("=== Importància de Variables No Psicològiques ===")
    calcular_importancia_xgboost(X_no_psicologiques, y, title="Importància de Variables No Psicològiques")

    # 5.2 Separació per sexe
    for sexe, label in [(1, "Homes"), (2, "Dones")]:
        df_sexe = df[df['sex'] == sexe]
        X_sexe = pd.DataFrame(scaler.fit_transform(df_sexe[variables_academiques + variables_numeriques + variables_binàries]),
                              columns=variables_academiques + variables_numeriques + variables_binàries)
        y_sexe = df_sexe['mental_health_status']
        print(f"=== Importància de Variables per Sexe: {label} ===")
        calcular_importancia_xgboost(X_sexe, y_sexe, title=f"Importància per Sexe: {label}")

    # 5.3 Clustering amb t-SNE i KMeans
    print("=== Clustering amb KMeans ===")
    tsne_clusters = tsne_clustering(X_psicologiques, n_clusters=3, method="KMeans")
    print("=== Clustering amb GMM ===")
    tsne_clusters = tsne_clustering(X_psicologiques, n_clusters=3, method="GMM")

    # 5.4 Densitat de variables psicològiques per clúster
    calcular_densitat_clusters(df, tsne_clusters['Cluster'], variables_psicologiques)

    # 5.5 Entrenar Regressió Logística
    print("=== Entrenament de Regressió Logística ===")
    entrenar_logistic_regression(X, y)
