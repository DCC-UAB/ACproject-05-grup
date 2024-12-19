'''
La finalitat d'aquest script és carregar el dataframe
final, on ja hi han totes les variables normalitzades
i estandaritzades segons calgui. Dediquem un fitxer a 
fer-ho per evitar redundància a la hora de reescriure 
el codi de la creació de dataframes.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_df():
    file = "dataset.csv"
    dataset = pd.read_csv(file)
    df = dataset.drop(['id', 'amsp'], axis=1)
    return df

def load_final():
    df = load_df()

    df_cat_order = df[['year', 'stud_h', 'health']]
    df_cat_norder = df[['glang','sex']]
    df_num = df[['age', 'jspe', 'qcae_cog', 'qcae_aff', 'erec_mean', 'mbi_ea']]
    df_bin = df[['part', 'job', 'psyt']]

    df_cat_norder_encoded = pd.get_dummies(df_cat_norder, columns=['glang', 'sex'], prefix=['glang', 'sex'])

    scaler = StandardScaler()
    scaler2 = StandardScaler()

    df_num_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)
    df_cat_order_scaled = pd.DataFrame(scaler2.fit_transform(df_cat_order), columns=df_cat_order.columns)
    df_final = pd.concat([df_cat_order_scaled, df_cat_norder_encoded, df_num_scaled, df_bin], axis=1)

    bool_columns = df_final.select_dtypes(include='bool').columns
    for col in bool_columns:
        df_final[col] = df_final[col].astype(int)
    
    return df_final


def load_max_scaled():
    df = load_df()
    df_max_scaled = df.copy() 
    for column in df_max_scaled.columns: 
        df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max() 
    return df_max_scaled

def load_min_max_scaled():
    df = load_df()
    df_min_max_scaled = df.copy() 
    for column in df_min_max_scaled.columns: 
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
    return df_min_max_scaled


