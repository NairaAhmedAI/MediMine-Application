import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import gridfs

# --- Step 1: Connect to MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["NHS_DB"]
conditions_col = db["conditions"]
fs = gridfs.GridFS(db)
models_meta = db["models_meta1"]

# --- Step 2: Load data from MongoDB ---
data = list(conditions_col.find({}))
df = pd.DataFrame(data)

# --- Step 3: TF-IDF features ---
tfidf_vectorizer_agg = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_agg = tfidf_vectorizer_agg.fit_transform(df["condition"]).toarray()

# --- Step 4: Agglomerative Clustering ---
n_clusters_agg = 60
agg_model = AgglomerativeClustering(
    n_clusters=n_clusters_agg,
    metric="cosine",
    linkage="average"
)
df["agg_cluster"] = agg_model.fit_predict(X_agg)

# --- Step 5: Simple evaluation metric ---
# cluster (intra-cluster similarity)


def cluster_cohesion(X, labels):
    scores = []
    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        if len(cluster_points) > 1:
            sims = cosine_similarity(cluster_points)
            upper = np.triu_indices_from(sims, k=1)
            scores.append(sims[upper].mean())
    return np.mean(scores) if scores else 0.0


cohesion = cluster_cohesion(X_agg, df["agg_cluster"].values)

print(f"✅ Agglomerative model trained | Cohesion score: {cohesion:.3f}")

# --- Step 6: Save model + vectorizer to GridFS ---
model_binary = pickle.dumps({
    "vectorizer": tfidf_vectorizer_agg,
    "clustering": agg_model
})
model_id = fs.put(model_binary, filename="agg_clustering.pkl")

# --- Step 7: Save metadata ---
models_meta.insert_one({
    "name": "agg_clustering_v1",
    "type": "AgglomerativeClustering",
    "gridfs_id": model_id,
    "labels": df["condition"].unique().tolist(),
    "metrics": {"cohesion": round(float(cohesion), 3)},
    "created": datetime.utcnow()
})

print("✅ Model + metadata saved to MongoDB")
