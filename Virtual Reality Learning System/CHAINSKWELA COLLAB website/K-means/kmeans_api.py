from flask import Flask, request, jsonify
from flask_cors import CORS
from K_means_Training_Model import predict_knowledge_level
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize Firebase Admin SDK (update the path to your service account key)
cred = credentials.Certificate('')
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/predict_knowledge', methods=['POST'])
def predict_knowledge():
    data = request.json
    user_id = data.get('user_id')
    count_1 = int(data.get('count_1', 0))
    count_3 = int(data.get('count_3', 0))
    count_5 = int(data.get('count_5', 0))

    # Let the model compute knowledge_level and total_points
    knowledge_level, total_points = predict_knowledge_level(count_1, count_3, count_5)

    # Store result in Firestore (collection: 'knowledge_results', document: user_id)
    if user_id:
        db.collection('knowledge_results').document(user_id).set({
            'count_1': count_1,
            'count_3': count_3,
            'count_5': count_5,
            'knowledge_level': knowledge_level,
            'total_points': total_points
        }, merge=True)
        # Also update the responses collection with the knowledge_level field
        db.collection('responses').document(user_id).set({
            'knowledge_level': knowledge_level
        }, merge=True)

    return jsonify({
        "message": "Good Job"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
