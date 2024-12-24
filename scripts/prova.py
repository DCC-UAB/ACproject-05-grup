import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ======================================
# 1. Càrrega de datasets guardats (pickle)
# ======================================
df_file = "pickles/df.pk1"
df = pd.read_pickle(df_file)

# ======================================
# 2. Funció: Importància de les variables amb Random Forest
# ======================================
def calcular_importancia_random_forest(X, y, title="Importància de Variables"):
    model = RandomForestClassifier(random_state=42)
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
# 3. Preprocessament i selecció de variables d'interès
# ======================================
variables_psicologiques = ['cesd', 'stai_t', 'mbi_ex']
variables_academiques = ['year', 'stud_h', 'health']
variables_binàries = ['part', 'job', 'psyt']
variables_numeriques = ['age', 'jspe', 'qcae_cog', 'qcae_aff', 'erec_mean', 'mbi_ea', 'mbi_cy']

# Crear índex de salut mental (mental_health_status)
df['mental_health_index'] = df[variables_psicologiques].mean(axis=1)
llindar = df['mental_health_index'].median()
df['mental_health_status'] = df['mental_health_index'].apply(lambda x: 1 if x > llindar else 0)

# Escalar les variables
scaler = StandardScaler()
X_psicologiques = pd.DataFrame(scaler.fit_transform(df[variables_psicologiques]), columns=variables_psicologiques)
X_no_psicologiques = pd.DataFrame(scaler.fit_transform(df[variables_academiques + variables_numeriques + variables_binàries]), 
                                  columns=variables_academiques + variables_numeriques + variables_binàries)

y = df['mental_health_status']

# Divisió train/test
X_train, X_test, y_train, y_test = train_test_split(X_psicologiques, y, test_size=0.2, random_state=42)

# ======================================
# 4. Importància de les variables psicològiques amb Random Forest
# ======================================
print("=== Importància de Variables Psicològiques ===")
importancia_psicologiques = calcular_importancia_random_forest(X_train, y_train, title="Importància de Variables Psicològiques")

# ======================================
# 5. Si vols també la importància de les variables no psicològiques:
# ======================================
X_train_no_psicologiques, X_test_no_psicologiques, y_train_no_psicologiques, y_test_no_psicologiques = train_test_split(
    X_no_psicologiques, y, test_size=0.2, random_state=42)

print("=== Importància de Variables No Psicològiques ===")
importancia_no_psicologiques = calcular_importancia_random_forest(X_train_no_psicologiques, y_train_no_psicologiques, 
                                                                   title="Importància de Variables No Psicològiques")
