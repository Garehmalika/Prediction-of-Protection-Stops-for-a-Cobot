# Par : GAREH Malika et BAMMAD Ihsane
# Prédiction des Arrêts de Protection d’un Cobot UR3

## Introduction

Ce projet est axé sur la prédiction des arrêts de protection (protective stops) d’un **cobot UR3** dans un environnement industriel en utilisant des modèles d’apprentissage automatique et de deep learning. L'objectif est d'exploiter des données de capteurs, telles que les courants, les températures et les vitesses des articulations du robot, pour prédire si un arrêt de protection va se produire dans les 10 prochaines unités de temps.

Le projet fait partie d'un projet de maintenance prédictive, visant à éviter les arrêts non planifiés des équipements dans un environnement industriel.

## Méthodologie Adoptée

1. **Prétraitement des données** : Nettoyer et transformer les données brutes issues du dataset "UR3 CobotOps" pour les rendre compatibles avec les modèles d’apprentissage automatique.
2. **Modélisation** :
   - Entraîner plusieurs modèles d'apprentissage automatique (LSTM, GRU, RNN, SVM, Transformer) pour prédire les arrêts de protection du cobot.
   - Comparer les performances des modèles en fonction de différentes métriques.
3. **Optimisation des hyperparamètres** : Tester et ajuster les hyperparamètres pour améliorer les performances des modèles.
4. **Déploiement du modèle** : Intégrer le modèle dans une API **Flask** permettant de faire des prédictions via une requête HTTP.
5. **Conteneurisation** : Dockeriser l’application Flask pour faciliter son déploiement en environnement de production.
## Hyperparamètres Testés

# LSTM (Long Short-Term Memory)
Le modèle LSTM a été configuré avec les hyperparamètres suivants :

Nombre de couches : 1 (Le modèle LSTM utilisé est composé d'une seule couche LSTM, contrairement à la version à deux couches mentionnée dans les tests initiaux).

Nombre de neurones par couche : 50 (Dans le code, la couche LSTM contient 50 neurones).

Activation : ReLU (La fonction d'activation utilisée pour la couche LSTM est ReLU, bien que tanh soit la fonction d'activation par défaut pour les LSTM).

Optimiseur : Adam (L'optimiseur Adam est utilisé pour entraîner le modèle).

Fonction de perte : Binary Crossentropy (La fonction de perte utilisée pour la classification binaire).

Batch Size : 32 (La taille des lots utilisée dans le générateur de séries temporelles pour l'entraînement et les tests).

Séquences d'entrée : 10 (Le modèle est alimenté avec des séquences de 10 pas de temps pour chaque exemple).

# GRU (Gated Recurrent Units)

Nombre de couches : 2

Nombre de neurones par couche : 32

La couche GRU contient 32 neurones, ce qui est un compromis entre la complexité du modèle et sa capacité à apprendre les séquences temporelles.

Activation : Par défaut, l'activation de la couche GRU est tanh, mais ici la couche GRU est utilisée avec Bidirectional, ce qui permet au modèle d'exploiter les informations à la fois dans le passé et dans le futur d'une séquence.

Optimiseur : Adam

Fonction de perte : Binary Crossentropy

Comme il s'agit d'une tâche de classification binaire (prédire si un arrêt de protection va se produire), la fonction de perte utilisée est binary_crossentropy, adaptée pour les problèmes de classification binaire.

Batch Size : 32

# RNN (Recurrent Neural Network)
Le modèle RNN a été configuré avec les hyperparamètres suivants :

Nombre de couches : 1

Le modèle RNN est composé d'une seule couche SimpleRNN. Cette couche est utilisée pour capturer les dépendances temporelles dans les données.

Nombre de neurones par couche : 64

La couche SimpleRNN contient 64 neurones, ce qui permet au modèle d'apprendre les relations temporelles dans les séquences de données.

Activation : tanh (Par défaut pour la couche SimpleRNN, bien que cela ne soit pas explicitement précisé, c'est la fonction d'activation utilisée dans les réseaux RNN de Keras).

Optimiseur : Adam

Fonction de perte : Binary Crossentropy

Batch Size : 32

Dropout : 0.2

Une couche Dropout est ajoutée après la couche RNN pour éviter le sur-apprentissage (overfitting). Cela permet de "désactiver" de manière aléatoire 20% des neurones pendant l'entraînement.

Séquences d'entrée : 10

# Transformer

Nombre de têtes d'attention : 4

Le modèle utilise 4 têtes d'attention dans la couche MultiHeadAttention. Cela permet au modèle de se concentrer sur différentes parties de la séquence d'entrée de manière parallèle, capturant ainsi plusieurs aspects des dépendances temporelles.

Dimension des clés et des valeurs (key_dim) : 64

La dimension des clés et des valeurs est fixée à 64.

Optimiseur : Adam

Fonction de perte : Binary Crossentropy

La binary_crossentropy est utilisée comme fonction de perte, adaptée pour des tâches de classification binaire où l'objectif est de prédire une sortie de type 0 ou 1.

Batch Size : 32

Dropout : 0.2

Un taux de dropout de 20% est appliqué après la normalisation pour éviter le sur-apprentissage (overfitting) du modèle.

Séquences d'entrée : 10

# SVM (Support Vector Machine)
Le modèle SVM a été configuré avec les hyperparamètres suivants :

Kernel : Linear (Le noyau linéaire est utilisé pour classifier les données).

Poids de classe : Balanced (Le modèle utilise l'option class_weight='balanced' pour gérer les classes déséquilibrées).

Probabilité : True (Active le calcul des probabilités pour la courbe ROC).

Optimiseur : Aucun nécessaire, car SVM utilise un algorithme interne pour l'optimisation

# Comparaison des modèles

LSTM a montré une meilleure capacité à capturer des dépendances à long terme en comparaison avec les autres modéles, mais il est plus coûteux en termes de ressources computationnelles.


## Prérequis

Pour exécuter ce projet, vous devez avoir installé les dépendances suivantes :

- **Python 3.11**
- **TensorFlow 2.7** pour le deep learning (LSTM, GRU, RNN, Transformer).
- **Scikit-learn** pour le modèle SVM et la gestion des données.
- **Numpy** et **Pandas** pour le traitement des données.
- **Matplotlib** et **Seaborn** pour la visualisation des résultats.

Vous pouvez installer les dépendances nécessaires en utilisant la commande suivante :

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn flask
```

## Installation

1. Clonez le dépôt depuis GitHub:
   
   ```bash
   git clone [[https://github.com/votre-utilisateur/projet-cobot-ur3.git](https://github.com/Garehmalika/Prediction-of-Protection-Stops-for-a-Cobot.git)](https://github.com/Garehmalika/Prediction-of-Protection-Stops-for-a-Cobot.git)
   cd projet-cobot-ur3
   ```

2. Installez les dépendances nécessaires en exécutant :

   ```bash
   pip install -r requirements.txt
   ```

3. Téléchargez le dataset "UR3 CobotOps" et placez-le dans le répertoire `data/`.

## Utilisation

Le projet contient plusieurs modèles d’apprentissage automatique pour prédire les arrêts de protection d'un cobot UR3 :

### LSTM (Long Short-Term Memory)

Le modèle LSTM est utilisé pour traiter des séquences temporelles et capturer les dépendances à long terme dans les données. Il est utilisé pour prédire si un arrêt de protection se produira dans les prochaines unités de temps, en fonction des données des 10 dernières unités.

### RNN (Recurrent Neural Network)

Le modèle RNN est un réseau neuronal récurrent classique qui permet également de traiter des séquences de données. Bien qu’il puisse être plus simple que LSTM, il peut avoir des limitations pour capturer des dépendances à long terme dans les données.

### SVM (Support Vector Machine)

Le modèle **SVM** est un classificateur qui peut être utilisé pour la classification binaire des arrêts de protection. Il est particulièrement utile pour des données linéaires ou peu bruitées.

### GRU (Gated Recurrent Units)

Le modèle **GRU** est une version plus légère de LSTM, souvent plus rapide à entraîner tout en maintenant des performances similaires. Il est également utilisé pour des données séquentielles et peut être plus adapté pour des problèmes avec moins de données ou des ressources limitées.

### Transformer

Le modèle **Transformer** utilise des mécanismes d’attention pour capturer les relations complexes dans les séries temporelles. Ce modèle est particulièrement puissant pour des séquences longues et peut surpasser les autres modèles de type RNN pour certaines tâches.

### API Flask

Le projet inclut également une API **Flask** permettant de charger un modèle entraîné et de faire des prédictions via une requête HTTP. Voici comment l'utiliser :

1. Lancez le serveur Flask :

   ```bash
   python app.py
   ```

2. Faites une requête HTTP POST pour obtenir des prédictions à partir du modèle, avec un exemple d'entrée sous forme de JSON.

### Conteneurisation avec Docker

Pour déployer l’application en environnement de production, l’application Flask est conteneurisée avec **Docker**. Voici les étapes pour exécuter le conteneur Docker :

1. **Construisez l'image Docker** :

   ```bash
   docker build -t cobot-api .
   ```

 2.**Exécutez le conteneur Docker** :

   ```bash
   docker run -p 5000:5000 cobot-api
   ```

L'API sera alors disponible sur `http://localhost:5000`.

## Tests et Évaluations

Les modèles sont évalués en utilisant plusieurs métriques de classification, notamment :

- **Accuracy** : Précision générale du modèle.
- **Précision** : Capacité du modèle à éviter les faux positifs.
- **Rappel** : Capacité du modèle à détecter les vrais positifs.
- **F1-score** : Moyenne harmonique de la précision et du rappel.

Les performances des différents modèles sont comparées et analysées pour déterminer lequel est le plus adapté à la tâche de prédiction des arrêts de protection du cobot.

## Structure du Code

Le projet est organisé comme suit :

```
/project-IA
    /data                  # Dossier contenant le dataset "UR3 CobotOps"
    /models                # Dossier contenant les modèles LSTM, RNN, SVM, GRU, Transformer
    /api                   # Code de l'API Flask pour le déploiement
    /docker                # Fichiers pour la conteneurisation avec Docker
    /notebooks             # Notebooks Jupyter pour l'exploration et le prétraitement des données
    /requirements.txt      # Liste des dépendances Python
    app.py                 # Fichier principal de l'API Flask
    Dockerfile             # Fichier pour créer l'image Docker
    README.md              # Documentation du projet
```





---

### Conclusion

Ce projet vise à développer un modèle d'intelligence artificielle pour prédire les arrêts de protection d’un cobot en utilisant plusieurs approches d'apprentissage automatique et deep learning. L’application est déployée via une API Flask et conteneurisée avec Docker pour une utilisation en production.

