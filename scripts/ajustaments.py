import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time

file = "dataset.csv"
dataset = pd.read_csv(file)
print(dataset.head())

df = dataset.drop(['id', 'amsp'], axis=1)

df_cat = df[['sex', 'year', 'glang', 'part', 'job', 'stud_h', 'health', 'psyt']]
df_num = df[['age', 'jspe', 'qcae_cog', 'qcae_aff', 'erec_mean', 'cesd', 'stai_t', 'mbi_ex', 'mbi_cy', 'mbi_ea']]
# df_cat = df[['sex', 'year', 'glang', 'part', 'job', 'stud_h', 'health', 'psyt']]
# df_num = df[['age', 'jspe', 'qcae_cog', 'qcae_aff', 'erec_mean']]

vars_categoriques = ['sex', 'year', 'glang', 'part', 'job', 'stud_h', 'health', 'psyt']
vars_num = ['age', 'jspe', 'qcae_cog', 'qcae_aff', 'erec_mean', 'cesd', 'stai_t', 'mbi_ex', 'mbi_cy', 'mbi_ea']
vars_bin = ['part', 'job', 'psyt']


#DETECCIÓ DE BIAIX
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
  
# apply normalization techniques 
for column in df_max_scaled.columns: 
    df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max() 

df_dis(df_max_scaled)

#NORMALITZACIÓ MIN-MAX
df_min_max_scaled = df.copy() 
  
# apply normalization techniques 
for column in df_min_max_scaled.columns: 
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())     
  
# view normalized data 
df_dis(df_min_max_scaled)

#COMPROVACIONS
df_default = df.copy()
df_default.head()

df_max_scaled.head()

df_min_max_scaled.head()