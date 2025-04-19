import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# Example data
data = [
    ['Visual'],
    ['Aural'],
    ['Read/Write'],
    ['Kinesthetic'],
    ['Visual', 'Aural'],
    ['Read/Write', 'Kinesthetic']
]

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Fit the model
mlb.fit(data)

# Save the model to a file
joblib.dump(mlb, 'mlb.joblib')

print("mlb.joblib has been saved successfully.")
