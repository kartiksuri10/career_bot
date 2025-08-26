from sklearn.cluster import KMeans
import joblib
from app.config import DATA_PATH
import numpy as np
# Choose a number of clusters (you can tune this later)
NUM_CLUSTERS = 10

job_embeddings_path = DATA_PATH / "job_embeddings.npy"

job_embeddings = np.load(job_embeddings_path)

# Fit KMeans on job embeddings
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
kmeans.fit(job_embeddings)

# Save the model for future use
joblib.dump(kmeans, DATA_PATH / "kmeans_model.joblib")

# Save cluster assignments for each job
np.save(DATA_PATH / "job_cluster_labels.npy", kmeans.labels_)
