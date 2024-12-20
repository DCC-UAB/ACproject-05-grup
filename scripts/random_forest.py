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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
y = scaler.fit_transform(df[['stai_t']])

# Dividir el conjunto de dades en train i test
X_train, X_test, y_train, y_test = train_test_split(df_final, y, test_size=0.2, random_state=42)

# Crear el model de Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar el model
rf.fit(X_train, y_train)

# Prediccions en el conjunt de prova
y_pred = rf.predict(X_test)

# Evaluació del model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Error cuadrátic mitjà (MSE):", mse)
print("Coeficient de determinació (R²):", r2)

# Obtenir les importancies de les característiques
importances = rf.feature_importances_
feature_names = df_final.columns

# Crear un gráfico de barras para visualizar las importancias
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Importància')
plt.title('Importancia de les variables en la predicció de les característiques psicològiques')
plt.show()
