import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import RandomOverSampler
import joblib

# Load dataset
CSV_FILE_PATH = r'VARK_output1.csv'
df = pd.read_csv(CSV_FILE_PATH)

# Parse multi-labels
df['Label Learning Style'] = df['Label Learning Style'].astype(str)
labels = df['Label Learning Style'].str.split(',')

# Features and label
X = df[['Count_V', 'Count_A', 'Count_R', 'Count_K']].values
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)

# Handle imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Split data
test_size = 30 / len(X_resampled)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=test_size, random_state=42
)

# Train multi-label model
clf = OneVsRestClassifier(RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
))
clf.fit(X_train, y_train)

# Save model and label binarizer
joblib.dump(clf, 'random_forest_model.joblib')
joblib.dump(mlb, 'mlb.joblib')
print('Model and label binarizer have been trained and saved.')

# Prediction function
def predict_learning_style(count_v, count_a, count_r, count_k):
    clf = joblib.load('random_forest_model.joblib')
    mlb = joblib.load('mlb.joblib')
    input_data = np.array([[count_v, count_a, count_r, count_k]])
    pred = clf.predict(input_data)
    labels = mlb.inverse_transform(pred)
    return list(labels[0])

# Test prediction
count_v = 10
count_a = 15
count_r = 15
count_k = 10
predicted_styles = predict_learning_style(count_v, count_a, count_r, count_k)
print("Predicted Learning Styles:", predicted_styles)
