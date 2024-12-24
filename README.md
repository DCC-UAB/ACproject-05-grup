[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/USx538Ll)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17348930&assignment_repo_type=AssignmentRepo)


Ilias Dachouri - 1673164 | Bernat Vidal - 1670982 | Joan Colillas - 1670247



# **Anàlisi i clusterització de la salut mental dels estudiants mitjançant Machine Learning**

## 1. Descripció del projecte
Aquest projecte té com l'objectiu d'analitzar l'estat de salut mental d'un grup d'estudiants mitjançant tècniques de Machine Learning. Utilitzarem algorismes de clusterització per agrupar els estudiants en funció de les seves característiques individuals, identificant així patrons i tendències originalment amagats que puguin ser indicatius de diferents estats mentals. A través d'aquesta anàlisi, es pretén identificar quins grups d'estudiants necessiten més atenció i suport, per tal de dissenyar intervencions més eficients per millorar la seva salut mental. Un cop acabat l'estudi, l'objectiu final és proporcionar recomanacions específiques per a la millora de la salut mental dels estudiants, amb un enfocament personalitzat en funció dels seus perfils.

## 2. Exploració, tractament i adaptació del Dataset
El [dataset](https://www.kaggle.com/code/faressayadi/medical-student-health-analysis-fares-sayadi/input) escollit  recull un conjunt de dades personals dels estudiants, com el seu gènere, variables relacionats amb el seu dia a dia, com poden ser el fet de si tenen parella o no, si treballen etc. juntament amb variables que descriuen diversos factors relacionats amb el seu estat mental, com el seu nivell d'ansietat o de depressió.

En concret tenim 886 registres després d'haver realitzat la neteja prèvia al dataset, i cada registre compta amb 20 variables, de les quals n'utilitzarem 18 per fer el clustering. S'han eliminat les variables de *id*, ja que no tenen cap rellevància per al clustering, i la variable *amsp*, ja que no es descriu enlloc de què tracta.

Dins el dataset ens trobem amb variables numèriques i categòriques ordinals i nominals amb diferents rangs de treball. Així doncs, hem cregut necesàri aplicar una série de modificacions per tal d'obtenir el millor resultat en el clustering.

Hem normalitzat i estandaritzat les **variables numèriques** i les **variables categòriques ordinals** (amb ordre) per tal d'assegur que tinguin el mateix rang i evitar que unes variables tinguin més influència que d'altres, fet el qual és especialment important en algoritmes de clustering (com Kmeans) ja que es basen en la distància euclidiana, i una diferència en l'escala pot afectar negativament als resultats.

Per tractar les **variables categòriques nominals** (sense ordre) hem fet servir un one-hot encoder per separar les seves categories en diferents variables binàries, de manera que cada cateogira es converteix en una nova columna amb valors de 0 o 1.

### **- Variables categòriques:**
* **sex**: gènere *[1 -> Home | 2 -> Dona | 3 -> No binari]*
* **year**: any acadèmic *[1 -> Bmed1 | 2 -> Bmed2 | 3 -> Bmed3 | 4 -> Mmed1 | 5 -> Mmed2 | 6 -> Mmed3]*
* **part**: parella *[1 -> Si | 0 -> No]*
* **job**:   treball *[1 -> Si | 0 -> No]*
* **psyt**:  s'ha consultat un psicòleg en els últims 12 mesos? *[0 -> No | 1 -> Si]*
* **glang**: llengua materna *[1 -> Francès | 15 -> Alemany | 53 -> Català...]*
* **health**: nivell de satisfacció amb la salut *[1 -> Molt Baix | 5 -> Molt alt]*
* **stud_h**: hores d'estudi per setmana

### **- Variables numèriques:**
* **age**: edat *[17 - 49]*
* **jspe**: nivell d'emapatía *[67 - 125]*
* **qcae_cog**: nivell cognitiu *[37 - 76]*
* **qcae_aff**: nivell d'afecció *[18 - 48]*
* **erec_mean**: percentatge total de respostes correctes en el GERT, un test on s'avalua si les persones poden reconeixer les emocions basant-se en llenguatge no verbal *[0.35 - 0.95]*
* **cesd**: escala de depressió *[0 - 56]*
* **stai_t**: escala d'ansietat *[20 - 77]*
* **mbi_ex**: cansament emocional *[5- 30]*
* **mbi_cy**: cinisme -> Mesura que tant distant una persona se sent respecte el seu voltant *[4 - 24]*
* **mbi_ea**: eficàcia acadèmica *[10 - 36]*

## 3. Preguntes formulades
Abans de l'estudi, ens hem formulat dues preguntes que pretenem respondre amb els resultats obtinguts, i en l'apartat de conclusions al finalitzar el projete:  

-  **Quins grups d'estudiants haurien de rebre major suport per a la seva salut mental?**

- **Quines característiques comparteixen els estudiants amb millor/pitjor estat de salut mental?**

## 4. Requeriments per l'execució i instruccions
El codi fa servir llibreries de Machine Learning i de processament de dades les quals necessiten estar instalades en el sistema per que el codi funcioni correctament. 
En concret es fa servir:
- numpy
- pandas 
- matplotlib
- seaborn
- scikit-learn
- umap

Es poden instal·lar amb les següents comandes:

```bash
[Windows ⊞]
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn

[Linux 🐧]
pip3 install numpy pandas matplotlib seaborn scikit-learn umap-learn
```

Per executar qualsevol codi dins del repositori:
```bash
[Windows ⊞]
python <fitxer>.py

[Linux 🐧]
python3 <fitxer>.py
```

Els fitxers executables es troben dins la carpeta 'scripts' en aquest repositori. No és necessari fer cap modificació adicional per al seu correcte funcionament, però és poden modificar els paràmetres si és que així es desitja.

Els dataframes estàn guardats en format 'pickle' dins la carpeta 'pickles' en aquest repositori, i són importats automàticament quan s'executen els scripts on són requerits. No obstant, també hi ha la opció de importar-los directament del dataset *dataset.csv*, descomentant les línies que indiqui el script. 

## 5. Algoritmes utilitzats
Per al nostre cas hem fet servir algoritmes de clusterització per ajuntar els estudiants en diferents grups segons les seves característiques. Així doncs, hem fet servir els següents algoritmes de clustering:
- Kmeans
- Gmm (Gaussian Mixture Model)
- Aglomerative clustering

Abans de realitzar el clustering, hem valorat quines variables són més rellevants per al model. Per a això, hem utilitzat algorismes de regressió que ens han permès identificar les variables amb més importància, assegurant-nos així que les característiques més significatives influencien el procés de clusterització. En concret:
- Random Forest Regression
- XGBoost (eXtreme Gradient Boosting)

## 6. Resultats
Per comprovar i classificar els clústers, hem fet la mitjana de les variables psicològiques de depressió, ansietat i cansament emocional en cada clúster, per després fer la mitjana de mitjanes i obtenir un llindar amb el qual poder separar els estudiants que presenten millors o pitjors estats de salut mental.

La clusterització amb l'algorisme de kmeans i amb el agglomerative han donat pràcticament el mateix resulat de clustering, comparting el mateix valor per a la millor k òptima, que en aqusest cas ha sigut 3. Els tres clústers es poden classificar en:
    - Clúster amb bon estat de salut mental
    - Clúster amb estat de salut mental estable
    - Clúster amb mal estat de salut mental

Amb l'algorisme de gmm hem obtingut un resultat diferent per a la k, en aquest cas de 5, ja que l'algorisme de GMM és més flexible i pot trobar grups amb formes diferents, no només esfèriques com K-means o l'aglomerative. Els resultats han donat que, 3 dels 5 clústers presenten un estat de salut mental pitjor en comparació als altres 2 clústers.

## 7. Conclusions
Després d'analitzar els tres algoritmes, podem concluïr en que els algoritmes de kmeans i agglomerative donen donen clústers més diferenciats entre sí, i per tant, amb característiques més identificables.
En quan a les preguntes que ens havíem formulat al principi, aquestes són les conclusions a les que arribem després de fer l'estudi.

**Quins grups d'estudiants haurien de rebre major suport per a la seva salut mental?**
Després d'haver fet l'estudi, podem veure que hi ha un grup d'estudiants que realment necessita suport per a la seva salut mental, i aquest grup és el clúster de gent que té, de mitjana, un coeficient major d'ansietat, d'estrès i de cansament emocional.

**Quines característiques comparteixen els estudiants amb millor/pitjor estat de salut mental?**
En ambdós algoritmes (kmeans i agglomerative) es pot veure que la mitjana dels usuaris del clúster amb pitjor estat de salut mental, entre d'altres, comparteixen unes característiques en comú. 
- **Situació sentimental:** Tenen parella, però la proporció no és majoritària (55% dels casos).
- **Poca activitat laboral:** Només un 43% dels estudiants del grup tenen treball remunerat.
- **Salut física acceptable:** Declaren tenir una salut física moderada, amb una puntuació mitjana de 3.34 sobre 6.
- **Poca atenció psicològica:** Només el 35% del grup visita regularment un psicòleg o un professional de la salut mental.
- **Baixa eficiència acadèmica:** Presenten una mitjana baixa d'eficiència acadèmica (19.49 sobre 36), fet que pot reflectir dificultats en la gestió de l'estrès acadèmic o altres factors emocionals.

Els usuari que tenen un millor estat de salut mental comparteixen les següents característiques:
- **Situació sentimental:** El 68% d’aquests estudiants tenen parella, cosa que podria indicar una possible font de suport emocional.
- **Activitat laboral moderada:** El 47% del grup té feina remunerada, una proporció lleugerament superior als altres clústers.
- **Bona salut física:** Declaren una mitjana de salut física de 4.07 sobre 6, per sobre dels altres grups analitzats.
- **Baixa freqüència de visites psicològiques:** Només el 17% dels estudiants visita regularment un psicòleg o professional de la salut mental, possiblement perquè no perceben una necessitat urgent d’ajuda.
- **Alta eficiència acadèmica:** Presenten una puntuació mitjana alta en eficiència acadèmica (26.66 sobre 36), indicant un bon rendiment i gestió dels estudis.

En conclusió, els clústers amb valors elevats en les variables psicològiques són els més vulnerables i requereixen suport prioritari, mentre que els estudiants amb millors resultats en salut física, ocupació, eficàcia acadèmica i menys símptomes psicològics mostren un estat més equilibrat i estable.

# 8. Contacte
- Joan Colillas - 1670247@uab.cat
- Ilias Dachouri - 1673164@uab.cat
- Bernat Vidal - 1670982@uab.cat
