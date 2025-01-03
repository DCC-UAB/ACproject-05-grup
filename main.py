'''
Aquest fitxer permet executar i visualitzar totes les funcionalitats principals del 
directòri.
'''
import os
import pandas as pd
from scripts import df_loaders
from scripts.plots import plot_tsne_clusters, plot_tsne_clusters_2D, plot_heatmap, plot_sorted_classified_clusters
from scripts.gmm import gmm
from scripts.agglomerative import agglomerative_clustering
from scripts.xgboost_ import calcular_importancia_xgboost
from scripts.kmeans_ import kmeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def clear():
    if os.name == 'nt':  #  Windows
        os.system('cls')
    else:  # Linux & macOS
        os.system('clear')


print('''
===========================================================      
======  Anàlisi i clusterització de la salut mental  ======
======  dels estudiants mitjançant Machine Learning  ======
===========================================================    
      ''')

if os.path.exists('pickles/df.pk1'):
    print("S'han trobat dataframes pickle, es carregaràn automàticament...")
    df = pd.read_pickle('pickles/df.pk1')
    df_max_scaled = pd.read_pickle('pickles/df_max_scaled.pk1')
    df_min_max_scaled = pd.read_pickle('pickles/df_min_max_scaled.pk1')
    df_final = pd.read_pickle('pickles/df_final.pk1')
    df_no_objectius = pd.read_pickle('pickles/df_no_objectius.pk1')
else:
    print("No s'han trobat fitxers pickles per carregar els dataframes. Es carregaràn de nou.")
    df = df_loaders.load_df()
    df_max_scaled = df_loaders.load_max_scaled()
    df_min_max_scaled = df_loaders.load_min_max_scaled()
    df_final = df_loaders.load_final()
    df_no_objectius = df_loaders.load_no_objectius()
print("Dataframes carregats")

# Funció per entrenar i avaluar models
def entrenar_i_avaluar_model(model, model_name, X_train, X_test, y_train, y_test):
    # Entrenar
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Mètriques
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== {model_name} ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Matriu de confusió
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    return {"Accuracy": model.score(X_test, y_test), "AUC": roc_auc}

match int(input('''
Escull una opció:
    1. Feature Importance
    2. Clustering
               -> ''')):
    case 1:
        match int(input('''
[Has escollit: Feature Importance]
    Escull quin algorisme vols fer servir:
        1. Random Forest
        2. XGBoost
                -> ''')):
            case 1: # Random Forest
                ...
            
            case 2: # XGBoost
                ...

    case 2:
        match int(input('''
[Has escollit: Clustering]
    Escull quin algorisme vols fer servir:
        1. Kmeans
        2. Gmm (Gaussian Mixture Model)
        3. Agglomerative Clustering
                -> ''')):
            case 1: # Kmeans
                c = kmeans(df_no_objectius, 8)
                k = int(input("La millor k és 3. Quina k vols fer servir per a les visualitzacions? -> "))

            case 2: # Gmm
                c = gmm(df_no_objectius, 8)
                k = int(input("La millor k és 5. Quina k vols fer servir per a les visualitzacions? -> "))

            case 3: # Agglomerative
                c = agglomerative_clustering(df_no_objectius, 8)
                k = int(input("La millor k és 3. Quina k vols fer servir per a les visualitzacions? -> "))
        if input("Vols visualitzar els clústers? S/n -> ") in ('S', 's'):
            plot_tsne_clusters_2D(df_no_objectius, c, k)
            plot_tsne_clusters(df_no_objectius, c, k)
        
        if input("Vols visualitzar les característiques de cada clúster? S/n -> ") in ('S', 's'):
            plot_sorted_classified_clusters(df, c, ['cesd', 'stai_t', 'mbi_ex'], k)
            l_features = []
            while True:
                clear()
                inp = input(f'''
Variables i característiques:
- Variables categòriques:
* sex: gènere [1 -> Home | 2 -> Dona | 3 -> No binari]
* year: any acadèmic [1 -> Bmed1 | 2 -> Bmed2 | 3 -> Bmed3 | 4 -> Mmed1 | 5 -> Mmed2 | 6 -> Mmed3]
* part: parella [1 -> Si | 0 -> No]
* job:   treball [1 -> Si | 0 -> No]
* psyt:  s'ha consultat un psicòleg en els últims 12 mesos? [0 -> No | 1 -> Si]
* health: nivell de satisfacció amb la salut [1 -> Molt Baix | 5 -> Molt alt]
* stud_h: hores d'estudi per setmana

- Variables numèriques:
* age: edat [17 - 49]
* jspe: nivell d'emapatía [67 - 125]
* qcae_cog: nivell cognitiu [37 - 76]
* qcae_aff: nivell d'afecció [18 - 48]
* erec_mean: percentatge total de respostes correctes en el GERT, un test on s'avalua si les persones poden reconeixer les emocions basant-se en llenguatge no verbal [0.35 - 0.95]
* cesd: escala de depressió [0 - 56]
* stai_t: escala d'ansietat [20 - 77]
* mbi_ex: cansament emocional [5- 30]
* mbi_cy: cinisme -> Mesura que tant distant una persona se sent respecte el seu voltant [4 - 24]
* mbi_ea: eficàcia acadèmica [10 - 36]

        Característiques seleccionades: {l_features}
        Variable a visualitzar (escriu 'end' per acabar) -> ''')
                if inp in df.columns and inp not in l_features:
                    l_features.append(inp)
                elif inp == 'end':
                    break
                else:
                    print(f"La opció [{inp}] no es troba en el dataset")
            
        plot_heatmap(df, df_min_max_scaled, c, l_features, k)


#millores de les opcions amb feature importance, classificacio i clustering
match int(input('''
Escull una opció:
    1. Feature Importance
    2. Clustering
    3. Classificació
               -> ''')):
    case 1:  # Feature Importance
        match int(input('''
[Has escollit: Feature Importance]
    Escull quin algorisme vols fer servir:
        1. Random Forest
        2. XGBoost
                -> ''')):
            case 1:  # Random Forest
                ...
            case 2:  # XGBoost
                ...

    case 2:  # Clustering
        match int(input('''
[Has escollit: Clustering]
    Escull quin algorisme vols fer servir:
        1. Kmeans
        2. Gmm (Gaussian Mixture Model)
        3. Agglomerative Clustering
                -> ''')):
            case 1:  # Kmeans
                c = kmeans(df_no_objectius, 8)
                k = int(input("La millor k és 3. Quina k vols fer servir per a les visualitzacions? -> "))
            case 2:  # Gmm
                c = gmm(df_no_objectius, 8)
                k = int(input("La millor k és 5. Quina k vols fer servir per a les visualitzacions? -> "))
            case 3:  # Agglomerative
                c = agglomerative_clustering(df_no_objectius, 8)
                k = int(input("La millor k és 3. Quina k vols fer servir per a les visualitzacions? -> "))
        if input("Vols visualitzar els clústers? S/n -> ") in ('S', 's'):
            plot_tsne_clusters_2D(df_no_objectius, c, k)
            plot_tsne_clusters(df_no_objectius, c, k)

        if input("Vols visualitzar les característiques de cada clúster? S/n -> ") in ('S', 's'):
            plot_sorted_classified_clusters(df, c, ['cesd', 'stai_t', 'mbi_ex'], k)
    
    case 3:  # Classificació
        # Preprocessament
        scaler = StandardScaler()
        target = ['cesd', 'stai_t', 'mbi_ex']
        X = df_no_objectius
        y = (df[target].mean(axis=1) > df[target].mean(axis=1).median()).astype(int)
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

        # Entrenament i avaluació de models
        models = {
            "Regressió Logística": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        }

        metrics = {}
        for name, model in models.items():
            metrics[name] = entrenar_i_avaluar_model(model, name, X_train, X_test, y_train, y_test)

        # Comparació de models
        metrics_df = pd.DataFrame(metrics).T
        print("\nComparació de mètriques:")
        print(metrics_df)

        # Gràfic comparatiu
        metrics_df.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
        plt.title("Comparació de Métriques entre Models")
        plt.xlabel("Model")
        plt.ylabel("Puntuació")
        plt.xticks(rotation=0)
        plt.legend(loc="lower right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

