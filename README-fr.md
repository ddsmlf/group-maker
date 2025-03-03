# Programme de formation de groupes

Ce programme permet de former des groupes en se basant les réponses d'un Google Form. Le script propose plusieurs méthodes d'optimisation : le **clustering hiérarchique**, le **recuit simulé** et le **groupement par affinité mutuelle**.

## Table des matières

1. [Prévention](#prévention)
2. [Utilisation du script](#utilisation-du-script)
3. [Matrice d'affinité](#matrice-daffinité)
4. [Méthodes de formation des groupes](#méthodes-de-formation-des-groupes)
   - [Clustering hiérarchique](#clustering-hiérarchique)
   - [Recuit simulé](#recuit-simulé)
   - [Groupement par affinité mutuelle](#groupement-par-affinité-mutuelle)
5. [Contributions](#contributions)
6. [Licence](#licence)

## Prévention

L'objectif de ce programme est de simplifier la constitution des groupes par affinités. Le script fourni utilise une matrice d'affinité pour représenter les préférences entre étudiants et propose différentes méthodes pour optimiser la formation de ces groupes. Cependant, malgrès la tentative de trouver des compris sur les algorithmes utilisé, l'utilisation d'algorithmes pour la formation de groupes comporte toujours des risques :

- **Élèves isolés** : Certains étudiants peuvent se retrouver isolés si leurs affinités avec les autres sont faibles.
- **Choix incohérents** : Les résultats peuvent parfois sembler incohérents ou injustes selon l'algorithme choisis pour car des compris doivent être fais et sont parfois difficilement automatisable.
- **Erreurs de développement** : Des bugs dans le script peuvent entraîner des répartitions incorrectes ou inefficaces. Il ets important de vérifier les résultats obtenus avec les personnes concernées.
- **Dépendance aux données** : La qualité des groupes formés dépend fortement de la précision et de la fiabilité des données d'affinité fournies. En cas de manque d'impliquation des étudiants ou de données falsifiées, les résultats peuvent être gravement affectés.

<span style="color:red; font-weight:bold;">
Il est donc important de vérifier et de valider les résultats obtenus par ces algorithmes et, si nécessaire, d'ajuster manuellement les groupes pour garantir une répartition équitable et efficace.
</span>


## Utilisation du script

### Prérequis

Avant d'exécuter le script, assurez-vous d’avoir installé les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
```

### Récupération des données

Les voeux des étudiants doivent être stockés dans un fichier CSV. Les noms des etudiants doivent toujours être au format nom-prenom. La dernière colonne du CSV devra toujours correspondre au nom-prenom de l'étudiant et l'avant dernière colonne devra correspondre à la liste des étudiants avec lesquels il souhaite être en groupe séparé par des espaces et par ordre de préférence (le premier étudiant étant celui avec lequel il préfère être en groupe). Un étudiant peut soumettre autant de voeux qu'il le souhaite.

pour obtenir facilement ce CSV il ets recommandé d'utilisé un Google Form tel que celui-ci : [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdwULMnEmbYJo17J9D_oSJdKUfqPLXAScK1BSbYCJyS2hoRyw/viewform?usp=header) et de récupérer les réponses sous forme de CSV grâce à Google Sheets.

### Exécution
Le script peut être exécuté via la ligne de commande en spécifiant le fichier contenant les voeux des étudiants :

```bash
python main.py input.csv 4 -a recuit -o groupes.csv -v -p
```

Les options disponibles pour `-a` ou `--algorithm` sont :
- `hierarchical` pour le clustering hiérarchique,
- `simulated_annealing` pour le recuit simulé,
- `affinity_grouping` pour le groupement par affinité mutuelle,
- `random` pour une répartition aléatoire.

Les autres options sont :
- `-o` ou `--output_file` pour spécifier le fichier de sortie (par défaut `groups.csv`),
- `-p` ou `--plot` pour afficher le graphe des préférences des étudiants,
- `-v` ou `--verbose` pour afficher des informations détaillées pendant l'exécution.

Une fois exécuté, le script génère un fichier CSV contenant la répartition optimale des étudiants.


## Matrice d'affinité

La **matrice d'affinité** est une représentation carrée où chaque élément `(i, j)` indique le degré de préférence ou d'affinité de l'étudiant `i` pour l'étudiant `j`. L'echelle dépend du nombre d'étudiant, le score est calculé par `rang - nombre_etudiants_total - 1`, (0 par défault). Ainsi une valeur élevée indique une forte affinité, tandis qu'une valeur faible indique une faible affinité.

**Exemple** :

|       | Étudiant A | Étudiant B | Étudiant C |
|-------|------------|------------|------------|
| **Étudiant A** |     0      |     8      |     5      |
| **Étudiant B** |     7      |     0      |     6      |
| **Étudiant C** |     4      |     9      |     0      |

Dans cet exemple, l'étudiant A a une affinité de 8 pour l'étudiant B et de 5 pour l'étudiant C.

## Méthodes de formation des groupes

Différentes méthodes d'optimisation sont proposées pour former des groupes d'étudiants en fonction de leurs affinités mutuelles. Chaque méthode a ses propres avantages et inconvénients, et peut être plus ou moins adaptée en fonction du contexte. 
Par défault, la méthode recommmandé est le **groupement par affinité** mutuelle qui comprend une gestion des étudiants isolés tout en maximisant les affinités entre les étudiants.

### Clustering hiérarchique

Le **clustering hiérarchique** est une méthode de regroupement qui cherche à créer une hiérarchie de clusters. Dans le contexte de ce script, il est légèrement adaptée, chaque étudiant commence comme un cluster individuel, et une paire d'étudiant maximisant l'affinité mutuelle est créer puis on ajoute progressivement les étudiants les plus compatibles jusqu'à atteindre la taille de groupe souhaitée. 

**Étapes de l'algorithme** :

1. **Initialisation** : Chaque étudiant est considéré comme un cluster individuel.
2. **Recherche de la meilleure paire** : Identifier la paire d'étudiants non assignés ayant la plus grande affinité mutuelle.
3. **Formation du groupe initial** : Créer un groupe avec cette paire et calculer son score.
4. **Ajout progressif** : Ajouter au groupe les étudiants les plus compatibles jusqu'à atteindre la taille de groupe souhaitée.
5. **Répétition** : Répéter les étapes 2 à 4 jusqu'à ce qu'il ne reste plus d'étudiants non assignés.
6. **Gestion des restants** : Les étudiants restants sont ajoutés à de nouveaux groupes s'ils sont en nombre suffisant, sinon ils sont répartis dans les groupes existants.

**Calcul du score d'un groupe** :

Le score d'un groupe est la somme des affinités entre tous les membres du groupe. 

**Avantages** :
<span style="color:green">
1. Respecte les affinités : Les groupes sont formés en maximisant la somme des points entre les membres.
2. Évolutif : On peut commencer avec des groupes de taille 2 et les étendre progressivement.
3. Approche gloutonne simple : En fusionnant les groupes qui ont le plus fort score interne, on assure une bonne répartition.
</span>

**Limites possibles** :
<span style="color:red">
1. Piège du local-optimum : Une approche purement gloutonne risque d’enfermer certains élèves dans de mauvais groupes.
2. Déséquilibre : Certains groupes pourraient être beaucoup plus optimaux que d'autres, créant des inégalités.
3. Absence de diversité : Si certains élèves sont très populaires, ils risquent d’être regroupés en premier, laissant les autres avec peu d'affinités.
</span>


### Recuit simulé

Le **recuit simulé** est une technique d'optimisation stochastique inspirée du processus de recuit en métallurgie, où un matériau est chauffé puis refroidi progressivement pour atteindre un état énergétique minimal. Cette méthode est utilisée pour éviter les minima locaux en acceptant occasionnellement des solutions moins optimales, avec une probabilité qui diminue au fil du temps. Elle ajoute une dimension aléatoire à la formation des groupes en permettant des échanges entre les étudiants.

**Étapes de l'algorithme** :

1. **Initialisation** : Générer une répartition aléatoire des étudiants en groupes de taille fixe.
2. **Échange aléatoire** : Sélectionner aléatoirement deux étudiants de deux groupes différents et les échanger.
3. **Calcul du score** : Calculer le score total de la nouvelle répartition des groupes.
4. **Comparaison des scores** : Si la nouvelle répartition améliore le score global, elle est acceptée. 
5. **Arrêt** : L'algorithme s'arrête lorsque le score globale devient très proche de 1 ou après un nombre maximal d'itérations.

**Avantages** :
<span style="color:green">
1. Exploration de l'espace de recherche : La méthode aléatoire permet d'éviter les minima locaux et d'explorer différentes solutions.
2. Adaptabilité : Le recuit simulé peut être ajusté pour accepter des solutions moins optimales au début,
3. Flexibilité : Les échanges aléatoires permettent de tester différentes combinaisons de groupes.
</span>

**Limites possibles** :
<span style="color:red">
1. Temps de calcul : Le recuit simulé peut nécessiter un grand nombre d'itérations pour converger vers une solution optimale.
2. Risque de stagnation : Si l'algorithme reste bloqué dans une région de l'espace de recherche, il peut ne pas converger vers une solution optimale.
3. Privilégie le score global : Le recuit simulé peut ne pas prendre en compte les affinités individuelles entre les étudiants. Des étudiants se classant mutuellement en premier peuvent être séparés au détriment du score global.
</span>

### Groupement par affinité mutuelle

Cette méthode consiste à former des groupes en maximisant les affinités mutuelles entre étudiants dès le départ. Elle fonctionne en sélectionnant les couples ou sous-groupes ayant les meilleures affinités avant de compléter progressivement les groupes.

**Étapes de l'algorithme** :

1. **Tri des affinités** : Former des groupes en partant des étudiants se classant mutuellement dans les N-1 premiers (N étant la taille du groupe).
2. **Récupération des étudints isolés** : Ajouter les étudiants demandés par aucun autre étudiant en priorité avec leurs voeux.
3. **Ajout des étudiants restants** : Créer des groupes d'un seul étudiant avec les étudiants restants.
4. **Optimisation des groupes** : Fusionner les groupes ayant le plus d'affinités mutuelles jusqu'à atteindre la taille de groupe souhaitée.

**Avantages** :
<span style="color:green">
1. Maximisation des affinités : Les étudiants mutuellement d'accord pour être ensemble sont regroupés en priorité.

2. Gestion des étudiants isolés : Les étudiants peux demandés sont regroupés en priorité pour éviter qu'ils se retrouvent isolés.

3. Équilibre des groupes : Les groupes sont formés de manière à équilibrer le score globale pour les étudiants qui n'ont pas de grande préférences.
</span>

**Limites possibles** :
<span style="color:red">
1. Risque d'isoler certains étudiants : Les étudiants ayant des affinités faibles avec les autres peuvent se retrouver isolés.
2. Manque de diversité : Les groupes formés peuvent manquer de diversité si la majorité des étudiants se classent mutuellement en premier.
</span>

## Contributions
- Version 1.0.0 : [Mélissa Colin](https://github.com/ddsmlf) : Développement du script initial avec les méthodes de clustering hiérarchique, de recuit simulé et de groupement par affinité mutuelle.