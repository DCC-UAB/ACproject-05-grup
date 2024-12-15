[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/USx538Ll)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17348930&assignment_repo_type=AssignmentRepo)

<h3>
Illias Dachouri - 
Bernat Vidal - 
Joan Colillas
</h3>
link dataset: https://www.kaggle.com/code/faressayadi/medical-student-health-analysis-fares-sayadi/input

# Informe sobre el projecte: Mental Health

## 1. Exploració del Dataset
Abans de començar amb el modelatge, cal entendre les dades. Algunes preguntes importants.

Quines variables conté el dataset? Quines són categòriques i quines són numèriques?

### Categòriques:
* sex (gènere 1 -> Home | 2 -> Dona | 3 -> No binari)
* year (any acadèmic)
* part (parella 1 -> Si | 0 -> No)
* job   (treball 1 -> Si | 0 -> No)    
* psyt  (s'ha consultat un psicòleg en els últims 12 mesos? 0 -> No | 1 -> Si)
* glang (llengua materna)
* health (nivel de satisfacció amb la salut) [1 -> Molt Baix || 5 -> Molt alt]
* stud_h (hores d'estudi per setmana)
### Numèriques:
* age (edat)
* jspe (nivell d'emapatía)
* qcae_cog (nivell cognitiu)
* qcae_aff (nivell d'afecció)
* amsp () ***
* erec_mean (percentatge total de respostes correctes en el GERT, un test on s'avalua si les persones poden reconeixer les emocions basant-se en llenguatge no verbal)
* cesd (escala de depressió)
* stai_t (escala d'ansietat) 
* mbi_ex (cansament emocional)
* mbi_cy (cinisme -> Mesura que tant distant una persona se sent respecte el seu voltant)
* mbi_ea (eficàcia acadèmica)

### Hi ha dades mancants o valors atípics? Si és així, com els gestionarem?

No hi ha dades mancants ni duplicades. En cas de valors atípics (p. ex., hores d'estudi molt altes), podrien ser tractats amb tècniques com el truncament basat en percentils.

### Hi ha equilibri entre les classes objectiu (si n’hi ha)? Per exemple, quants estudiants mostren símptomes de malalties mentals en comparació amb aquells que no en mostren?
Les classes binàries, com psyt (presència de problemes psicològics), tenen una distribució desequilibrada. Aquest desequilibri s’hauria de considerar durant el modelatge supervisat per evitar biaixos.

### Com es distribueixen les variables rellevants? (p. ex., nivells d'estrès, hores de son, etc.)

cesd (depressió): Mitjana de 18, amb un rang entre 0 i 56.
stai_t (ansietat): Mitjana de 42, amb un rang ampli, indicant diversitat en nivells d'ansietat.
Hores d'estudi (stud_h): Mitjana de 25 hores, però amb variabilitat significativa (0 a 70 hores).

### Hi ha correlacions clares entre algunes variables i els possibles indicadors de malalties mentals?

Les correlacions entre cesd (depressió), stai_t (ansietat) i altres indicadors com mbi_ex (esgotament emocional)

## 2. Preparació del Dataset
Per aplicar tècniques de clustering o altres models, hem de netejar i estructurar les dades:

### Quines característiques són més rellevants per predir símptomes mentals? (Estratègies: anàlisi de correlació, selecció de característiques).

Variables com cesd, stai_t, mbi_ex i stud_h són fortes candidates basades en la seva relació amb l'estrès, l'ansietat i altres factors psicològics.

### Cal realitzar transformacions o normalitzacions de les dades?

Algunes variables, com cesd i stud_h, tenen rangs molt amplis i es beneficiarien de la normalització per a models com el k-means.

### Com gestionem les variables categòriques (p. ex., codificació One-Hot o embeddings)?

Codificació One-Hot per variables com sex i job. Això permetrà que els models tractin aquestes dades categòriques.

# Preguntes prèvies a l'estudi

Abans d'agrupar als estudiants basant-nos en el seu estat de salut mental, ens hem fet una sèrie de preguntes que volem contestar fent el clustering. Entre d'altres, ens hem plantejat les següents:

### Quin tipus de grups d'estudiants haurien de rebre major suport per la seva salut mental?

### Quines característiques comparteixen els estudiants amb millor/pitjor estat de salut mental?

### Com influeixen diferents característiques relacionades amb els estudis a la salut mental dels estudiants? 

### L'anàlisi estadístic es pot confirmar amb els resultats del clustering?