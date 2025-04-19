import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from sklearn.preprocessing import MultiLabelBinarizer

# Load dataset
CSV_FILE_PATH = r'VARK_output1.csv'  # Use local file
df = pd.read_csv(CSV_FILE_PATH)

# Multi-label: split labels by comma
df['Label Learning Style'] = df['Label Learning Style'].astype(str)
labels = df['Label Learning Style'].str.split(',')

# Features and label
X = df[['Count_V', 'Count_A', 'Count_R', 'Count_K']]
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)
label_classes = mlb.classes_

# For oversampling, use single-label (first label per sample)
single_label_y = [lbls[0] for lbls in labels]

# Oversample to handle class imbalance (single-label only)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, single_label_y)
data_resampled = pd.DataFrame(X_resampled, columns=['Count_V', 'Count_A', 'Count_R', 'Count_K'])
data_resampled['Label Learning Style'] = y_resampled

# Discretize counts into 3 bins
for col in ['Count_V', 'Count_A', 'Count_R', 'Count_K']:
    data_resampled[col] = pd.cut(data_resampled[col], bins=3, labels=['low', 'medium', 'high'])

# Split into train/test ensuring 30 test samples
test_size = 30 / len(data_resampled)
train_data, test_data = train_test_split(data_resampled, test_size=test_size, random_state=42)

# Define Bayesian Network structure
model = DiscreteBayesianNetwork([
    ('Count_V', 'Label Learning Style'),
    ('Count_A', 'Label Learning Style'),
    ('Count_R', 'Label Learning Style'),
    ('Count_K', 'Label Learning Style'),
])

# Train the model using Bayesian Estimator
model.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

# Create inference engine using VariableElimination
infer = VariableElimination(model)

# Prediction helper using query()
def predict_bn(row):
    evidence = {
        'Count_V': row['Count_V'],
        'Count_A': row['Count_A'],
        'Count_R': row['Count_R'],
        'Count_K': row['Count_K']
    }
    result = infer.query(variables=['Label Learning Style'], evidence=evidence, show_progress=False)
    prob_dist = result.values
    label_states = result.state_names['Label Learning Style']
    # Multi-label: return all labels with max probability (handle ties)
    max_prob = np.max(prob_dist)
    return ','.join([label_states[i] for i, p in enumerate(prob_dist) if p == max_prob])

# Run predictions on test set
y_test_true = test_data['Label Learning Style'].values
y_test_pred = test_data.apply(predict_bn, axis=1).values

# Evaluation
accuracy = accuracy_score(y_test_true, y_test_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test_true, y_test_pred)
print("Confusion Matrix:\n", conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test_true), yticklabels=np.unique(y_test_true))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", classification_report(y_test_true, y_test_pred))

# Save model, inference engine, and label binarizer
joblib.dump(model, 'bayesian_network_model.pkl')
joblib.dump(infer, 'bayesian_network_infer.pkl')
joblib.dump(mlb, 'mlb.pkl')
print("Bayesian Network model, inference engine, and label binarizer saved.")

# Prediction function for new inputs (multi-label)
def predict_learning_style(count_v, count_a, count_r, count_k):
    infer = joblib.load('bayesian_network_infer.pkl')
    mlb = joblib.load('mlb.pkl')

    def discretize(val):
        if val <= 3:
            return 'low'
        elif val <= 6:
            return 'medium'
        else:
            return 'high'

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
    # Return all labels with max probability (multi-label)
    return [label_states[i] for i, p in enumerate(prob_dist) if p == max_prob]

# Example usage
count_v = 5
count_a = 3
count_r = 7
count_k = 6
predicted_styles = predict_learning_style(count_v, count_a, count_r, count_k)
print("Predicted Learning Styles:", predicted_styles)
