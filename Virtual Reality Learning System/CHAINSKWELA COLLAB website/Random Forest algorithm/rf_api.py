from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'random_forest_model.joblib'
MLB_PATH = 'mlb.joblib'
clf = joblib.load(MODEL_PATH)
mlb = joblib.load(MLB_PATH)

# Initialize Firebase Admin SDK (use your own service account key path)
FIREBASE_CRED_PATH = r''  # Use your existing keyelative path
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/predict_vark', methods=['POST'])
def predict_vark():
    data = request.get_json()
    print(f"Received data: {data}")  # Debug: see what is received
    user_id = data.get('user_id')
    count_v = data.get('Count_V')
    count_a = data.get('Count_A')
    count_r = data.get('Count_R')
    count_k = data.get('Count_K')
    
    if None in (user_id, count_v, count_a, count_r, count_k):
        return jsonify({'error': 'Missing input'}), 400
    
    input_data = np.array([[count_v, count_a, count_r, count_k]])
    pred = clf.predict(input_data)
    print(f"Prediction result: {pred}")  # Debugging

    # Directly use the prediction result since it's a single label
    predicted_style = pred[0]  # pred is an array, take the first element

    # Attempt to store in Firestore
    try:
        db.collection('vark_knowledge_level').document(user_id).set({
            'Count_V': count_v,
            'Count_A': count_a,
            'Count_R': count_r,
            'Count_K': count_k,
            'predicted_style': predicted_style
        }, merge=True)
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        return jsonify({'error': 'Failed to save data to Firestore'}), 500

    return jsonify({'predicted_style': predicted_style})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
