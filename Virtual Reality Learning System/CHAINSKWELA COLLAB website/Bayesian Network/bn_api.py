from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'bayesian_network_model.pkl'
INFER_PATH = 'bayesian_network_infer.pkl'
MLB_PATH = 'mlb.pkl'
infer = joblib.load(INFER_PATH)
mlb = joblib.load(MLB_PATH)

FIREBASE_CRED_PATH = r'c:\Users\grafr\Thesis files\CHAINSKWELA COLLAB website\K-means\chainskwela-collab-firebase-adminsdk-fbsvc-b02cceb1bd.json'
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

def discretize(val):
    if val <= 3:
        return 'low'
    elif val <= 6:
        return 'medium'
    else:
        return 'high'

@app.route('/predict_vark_bn', methods=['POST'])
def predict_vark_bn():
    data = request.get_json()
    user_id = data.get('user_id')
    count_v = data.get('Count_V')
    count_a = data.get('Count_A')
    count_r = data.get('Count_R')
    count_k = data.get('Count_K')
    if None in (user_id, count_v, count_a, count_r, count_k):
        return jsonify({'error': 'Missing input'}), 400

    evidence = {
        'Count_V': discretize(count_v),
        'Count_A': discretize(count_a),
        'Count_R': discretize(count_r),
        'Count_K': discretize(count_k)
    }
    result = infer.query(variables=['Label Learning Style'], evidence=evidence, show_progress=False)
    prob_dist = result.values
    label_states = result.state_names['Label Learning Style']
    max_prob = np.max(prob_dist)
    predicted_styles = [label_states[i] for i, p in enumerate(prob_dist) if p == max_prob]
    predicted_styles_str = ','.join(predicted_styles)

    db.collection('vark_knowledge_level').document(user_id).set({
        'Count_V': count_v,
        'Count_A': count_a,
        'Count_R': count_r,
        'Count_K': count_k,
        'predicted_style': predicted_styles_str
    }, merge=True)

    return jsonify({'predicted_style': predicted_styles_str})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
