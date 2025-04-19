import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load your updated CSV data
data = pd.read_csv("KL_output_with_labels.csv")
data['Weighted_1'] = data['Count_1'] * 1
data['Weighted_3'] = data['Count_3'] * 3
data['Weighted_5'] = data['Count_5'] * 5

# Compute features for clustering
data['Total_Correct'] = data['Count_1'] + data['Count_3'] + data['Count_5']
data['Total_Weighted_Score'] = data['Weighted_1'] + data['Weighted_3'] + data['Weighted_5']
X_final = data[['Total_Correct', 'Total_Weighted_Score']]

# Standardize features
final_scaler = StandardScaler()
X_final_scaled = final_scaler.fit_transform(X_final)

# Train K-means
final_kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
final_kmeans.fit(X_final_scaled)

# Save the trained model and scaler
joblib.dump(final_kmeans, 'final_kmeans_model.pkl')
joblib.dump(final_scaler, 'final_scaler.pkl')

# Optional: Print cluster centers for analysis
centers = final_scaler.inverse_transform(final_kmeans.cluster_centers_)
print("Cluster centers (Total_Correct, Total_Weighted_Score):")
print(centers)
print("Cluster sizes:", pd.Series(final_kmeans.labels_).value_counts().to_dict())

print("Retraining complete. Model and scaler updated.")
