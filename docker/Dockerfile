# Étape 1 : Utiliser une image de base Python
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail
WORKDIR /app



# Étape 4 : Installer les dépendances à partir du fichier requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Copier les autres fichiers de l'application dans le conteneur
COPY . /app

# Étape 6 : Exposer le port 5000 (par défaut utilisé par Flask)
EXPOSE 5000

# Étape 7 : Lancer l'application Flask avec Gunicorn (serveur de production)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "api.app:app"]
