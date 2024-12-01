[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/USx538Ll)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17348930&assignment_repo_type=AssignmentRepo)

Informe sobre el projecte de Mental Health:

1. Exploració del Dataset
Abans de començar amb el modelatge, cal entendre les dades. Algunes preguntes importants:

· Quines variables conté el dataset? Quines són categòriques i quines són numèriques?

A continuació, un breu resum de les variables més rellevants:
Variables identificadores o generals: id, age, year, sex.
Indicadors de salut i benestar: health (estat de salut), psyt (problemes psicològics), cesd (escala de depressió), stai_t (nivell d'ansietat).
Indicadors d'hàbits i rendiment: stud_h (hores d'estudi), erec_mean (temps de recuperació).
Indicadors professionals: job, part (relacions laborals).
Mesures psicològiques: qcae_cog (empatia cognitiva), qcae_aff (empatia afectiva), amsp, mbi_ex (esgotament emocional), mbi_cy (cínicisme), mbi_ea (realització personal).

Variables categòriques: sex (gènere), part (parella), job (treball). Aquestes tenen un nombre limitat de valors únics.
Variables numèriques: Inclouen indicadors de salut (health, cesd, stai_t), hàbits (stud_h), i mètriques psicològiques (qcae_cog, mbi_ex, etc.).

· Hi ha dades mancants o valors atípics? Si és així, com els gestionarem?

No hi ha dades mancants ni duplicades. En cas de valors atípics (p. ex., hores d'estudi molt altes), podrien ser tractats amb tècniques com el truncament basat en percentils.

· Hi ha equilibri entre les classes objectiu (si n’hi ha)? Per exemple, quants estudiants mostren símptomes de malalties mentals en comparació amb aquells que no en mostren?
Les classes binàries, com psyt (presència de problemes psicològics), tenen una distribució desequilibrada. Aquest desequilibri s’hauria de considerar durant el modelatge supervisat per evitar biaixos.

· Com es distribueixen les variables rellevants? (p. ex., nivells d'estrès, hores de son, etc.)

cesd (depressió): Mitjana de 18, amb un rang entre 0 i 56.
stai_t (ansietat): Mitjana de 42, amb un rang ampli, indicant diversitat en nivells d'ansietat.
Hores d'estudi (stud_h): Mitjana de 25 hores, però amb variabilitat significativa (0 a 70 hores).

· Hi ha correlacions clares entre algunes variables i els possibles indicadors de malalties mentals?

Les correlacions entre cesd (depressió), stai_t (ansietat) i altres indicadors com mbi_ex (esgotament emocional)

2. Preparació del Dataset
Per aplicar tècniques de clustering o altres models, hem de netejar i estructurar les dades:

· Quines característiques són més rellevants per predir símptomes mentals? (Estratègies: anàlisi de correlació, selecció de característiques).

Variables com cesd, stai_t, mbi_ex i stud_h són fortes candidates basades en la seva relació amb l'estrès, l'ansietat i altres factors psicològics.

· Cal realitzar transformacions o normalitzacions de les dades?

Algunes variables, com cesd i stud_h, tenen rangs molt amplis i es beneficiarien de la normalització per a models com el k-means.

· Com gestionem les variables categòriques (p. ex., codificació One-Hot o embeddings)?

Codificació One-Hot per variables com sex i job. Això permetrà que els models tractin aquestes dades categòriques.

