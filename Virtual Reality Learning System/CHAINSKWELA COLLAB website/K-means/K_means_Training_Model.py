import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
data = pd.read_csv("KL_output_with_labels.csv")
data['Weighted_1'] = data['Count_1'] * 1
data['Weighted_3'] = data['Count_3'] * 3
data['Weighted_5'] = data['Count_5'] * 5
pairs = [
    ('Count_1', 'Weighted_1'),
    ('Count_3', 'Weighted_3'),
    ('Count_5', 'Weighted_5')
]
kmeans_models = {}
scalers = {}

for count_col, weighted_col in pairs:
    X = data[[count_col, weighted_col]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scalers[count_col] = scaler
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(X_scaled)
    data[f'Cluster_{count_col}'] = kmeans.labels_
    kmeans_models[count_col] = kmeans

data['Total_Correct'] = data['Count_1'] + data['Count_3'] + data['Count_5']
data['Total_Weighted_Score'] = data['Weighted_1'] + data['Weighted_3'] + data['Weighted_5']
X_final = data[['Total_Correct', 'Total_Weighted_Score']]
final_scaler = StandardScaler()
X_final_scaled = final_scaler.fit_transform(X_final)
final_kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
final_kmeans.fit(X_final_scaled)
data['Final_Cluster'] = final_kmeans.labels_
final_cluster_mapping = {0: 'Intermediate', 1: 'Beginner', 2: 'Difficult'}
data['Final_Category'] = data['Final_Cluster'].map(final_cluster_mapping)

# Save models
joblib.dump(kmeans_models, 'kmeans_models.pkl')
joblib.dump(scalers, 'scalers.pkl')
joblib.dump(final_kmeans, 'final_kmeans_model.pkl')
joblib.dump(final_scaler, 'final_scaler.pkl')

# Function for API/backend use
def predict_knowledge_level(count_1, count_3, count_5):
    """
    Predicts the knowledge level using the trained K-means model.
    The mapping of clusters to levels is determined by the average total_points of each cluster center.
    Returns:
        tuple: (knowledge_level, total_points)
    """
    import numpy as np
    import joblib

    # Compute features
    total_correct = count_1 + count_3 + count_5
    total_points = count_1 * 1 + count_3 * 3 + count_5 * 5
    X_input = np.array([[total_correct, total_points]])

    # Load trained models
    final_kmeans = joblib.load('final_kmeans_model.pkl')
    final_scaler = joblib.load('final_scaler.pkl')

    # Get cluster centers in original scale
    centers = final_scaler.inverse_transform(final_kmeans.cluster_centers_)
    # Sort clusters by total_points (second column)
    sorted_indices = np.argsort(centers[:, 1])
    # Map sorted cluster indices to levels
    cluster_to_level = {}
    cluster_to_level[sorted_indices[0]] = "Beginner"
    cluster_to_level[sorted_indices[1]] = "Intermediate"
    cluster_to_level[sorted_indices[2]] = "Advanced"

    # Scale input and predict cluster
    X_scaled = final_scaler.transform(X_input)
    cluster = final_kmeans.predict(X_scaled)[0]
    knowledge_level = cluster_to_level.get(cluster, "Unknown")

    return knowledge_level, total_points

# Only run plotting code when executed directly
if __name__ == "__main__":
    for count_col, weighted_col in pairs:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=count_col, y=weighted_col, hue=f'Cluster_{count_col}', data=data, palette='viridis')
        plt.scatter(scalers[count_col].inverse_transform(kmeans_models[count_col].cluster_centers_)[:, 0],
                    scalers[count_col].inverse_transform(kmeans_models[count_col].cluster_centers_)[:, 1],
                    s=300, c='red', label='Centroids')
        plt.xlabel(count_col)
        plt.ylabel(weighted_col)
        plt.title(f'K-Means Clustering with {count_col} and {weighted_col}')
        plt.legend()
        plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Total_Correct', y='Total_Weighted_Score', hue='Final_Category', data=data, palette='viridis')
    plt.scatter(final_scaler.inverse_transform(final_kmeans.cluster_centers_)[:, 0],
                final_scaler.inverse_transform(final_kmeans.cluster_centers_)[:, 1],
                s=300, c='red', label='Centroids')
    plt.xlabel('Total Correct')
    plt.ylabel('Total Weighted Score')
    plt.title('Final K-Means Clustering with Total Correct Answers and Weighted Scores')
    plt.legend()
    plt.show()
