'''
En aquest script es mostra el codi per normalitzar les dades dins d'un dataframe.
A més, un cop normalitzades, també es dona l'opció de mostrar les distribucions i
els valors/rangs de les variables en un plot.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os


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

# Carregar dataset guardat pickle
file = "pickles/dfs/df.pk1"
df = pd.read_pickle(file)

# ===============================================================================================

# Determine grid size
def df_dis(df):
    num_cols = len(df.columns)
    num_rows = math.ceil(num_cols / 4)  # Adjust columns per row as needed

    fig, axes = plt.subplots(num_rows, 4, figsize=(15, num_rows * 5))  # 3 columns per row
    axes = axes.flatten()

    # Loop through each column and plot on a specific subplot
    for i, column in enumerate(df.columns):
        ax = axes[i]
        if df[column].dtype == 'object':
            sns.countplot(x=column, data=df, ax=ax, palette="Set2")
            ax.set_title(f'Distribution of {column}')
            ax.tick_params(axis='x', rotation=45)
        else:
            sns.histplot(df[column], kde=True, ax=ax, color="skyblue")
            ax.set_title(f'Distribution of {column}')
            
    # Hide any unused subplots
    for i in range(num_cols, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

df_dis(df)


# NORMALITZACIÓ MAX
df_max_scaled = df.copy() 
for column in df_max_scaled.columns: 
    df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max() 

df_dis(df_max_scaled)

#NORMALITZACIÓ MIN-MAX
df_min_max_scaled = df.copy() 
for column in df_min_max_scaled.columns: 
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())     
  
df_dis(df_min_max_scaled)

# #COMPROVACIONS
# df_default = df.copy()
# df_default.head()
# df_max_scaled.head()
# df_min_max_scaled.head()