from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Charger uniquement le modèle LSTM et le scaler
lstm_model = load('Models/LSTM.pkl')  # Utilise load_model pour le modèle LSTM
scaler = load('Models/Scaler.pkl')

@app.route('/', methods=['GET'])
def home():
    return "Bienvenue sur l'API de prédiction LSTM avec succès !", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Vérifier si les champs 'input' sont présents dans la requête
    if 'input' not in data:
        return jsonify({'error': 'Le champ "input" est requis.'}), 400

    input_data = np.array(data['input']).reshape(1, -1)
    
    # Appliquer la transformation (scaling) des données d'entrée
    input_scaled = scaler.transform(input_data)

    # Prédiction avec le modèle LSTM
    prediction = lstm_model.predict(input_scaled)

    # Si la prédiction donne une probabilité, la convertir en 0 ou 1
    if prediction.shape[-1] == 1:
        prediction = (prediction > 0.5).astype(int)

    return jsonify({'prediction': int(prediction[0][0])})

@app.route('/test', methods=['GET'])
def test():
    return "API opérationnelle pour le modèle LSTM ✔️"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
