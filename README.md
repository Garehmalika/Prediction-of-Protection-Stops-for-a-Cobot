# Prédiction des Arrêts de Protection d’un Cobot UR3

## Introduction

Ce projet est axé sur la prédiction des arrêts de protection (protective stops) d’un **cobot UR3** dans un environnement industriel en utilisant des modèles d’apprentissage automatique et de deep learning. L'objectif est d'exploiter des données de capteurs, telles que les courants, les températures et les vitesses des articulations du robot, pour prédire si un arrêt de protection va se produire dans les 10 prochaines unités de temps.

Le projet fait partie d'un projet de maintenance prédictive, visant à éviter les arrêts non planifiés des équipements dans un environnement industriel.

## Objectifs du Projet

Les objectifs du projet incluent :
1. **Prétraitement des données** : Nettoyer et transformer les données brutes issues du dataset "UR3 CobotOps" pour les rendre compatibles avec les modèles d’apprentissage automatique.
2. **Modélisation** :
   - Entraîner plusieurs modèles d'apprentissage automatique (LSTM, GRU, RNN, SVM, Transformer) pour prédire les arrêts de protection du cobot.
   - Comparer les performances des modèles en fonction de différentes métriques.
3. **Optimisation des hyperparamètres** : Tester et ajuster les hyperparamètres pour améliorer les performances des modèles.
4. **Déploiement du modèle** : Intégrer le modèle dans une API **Flask** permettant de faire des prédictions via une requête HTTP.
5. **Conteneurisation** : Dockeriser l’application Flask pour faciliter son déploiement en environnement de production.

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

1. Clonez le dépôt depuis GitHub ou GitLab :
   
   ```bash
   git clone [https://github.com/votre-utilisateur/projet-cobot-ur3.git](https://github.com/Garehmalika/Prediction-of-Protection-Stops-for-a-Cobot.git)
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

Pour déployer l’application en environnement de production, l’application Flask est conteneurisée avec **Docker**. Voici les étapes pour construire et exécuter le conteneur Docker :

1. **Construisez l'image Docker** :

   ```bash
   docker build -t cobot-api .
   ```

2. **Exécutez le conteneur Docker** :

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

Si vous avez des questions ou des problèmes avec l'exécution du projet, n'hésitez pas à ouvrir une issue ou à me contacter.

