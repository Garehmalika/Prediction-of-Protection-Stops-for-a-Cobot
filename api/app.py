from flask import Flask, request, jsonify
from joblib import load

import numpy as np

app = Flask(__name__)

# Charger les modèles
cnn_model = load('Models/CNN.pkl')
lstm_model = load('Models/LSTM.pkl')  # Utilise load_model pour le modèle LSTM
gru_model = load('Models/Gru_weights.pkl')
scaler = load('Models/Scaler.pkl')

@app.route('/', methods=['GET'])
def home():
    return "Bienvenue sur l'API de prédiction avec succès !", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'input' not in data or 'model_type' not in data:
        return jsonify({'error': 'Les champs "input" et "model_type" sont requis.'}), 400

    input_data = np.array(data['input']).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    model_type = data['model_type'].lower()

    # Vérification du type du modèle avant d'appeler predict
    model = None
    if model_type == 'cnn':
        model = cnn_model
    elif model_type == 'lstm':
        model = lstm_model
    elif model_type == 'gru':
        model = gru_model
    else:
        return jsonify({'error': f"Type de modèle '{model_type}' non supporté. Choisissez parmi 'cnn', 'lstm', ou 'gru'."}), 400

    # Vérification si le modèle a la méthode 'predict'
    if not hasattr(model, 'predict'):
        return jsonify({'error': f"Le modèle '{model_type}' n'a pas de méthode 'predict'."}), 400

    # Prédiction avec le modèle sélectionné
    prediction = model.predict(input_scaled)

    # Si la prédiction donne une probabilité, on la convertit en 0 ou 1
    if prediction.shape[-1] == 1:
        prediction = (prediction > 0.5).astype(int)

    return jsonify({'prediction': int(prediction[0][0])})

@app.route('/test', methods=['GET'])
def test():
    return "API opérationnelle ✔️"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
