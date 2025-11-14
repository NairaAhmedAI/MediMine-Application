import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load TF-IDF vectorizer
# -----------------------------
try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer_agg = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load vectorizer.pkl: {e}")
    st.stop()

# -----------------------------
# Load pre-trained Agglomerative Clustering model
# -----------------------------
try:
    with open("agg_model.pkl", "rb") as f:
        agg_model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load agg_model.pkl: {e}")
    st.stop()

# -----------------------------
# Load Disease DataFrame from pickle
# -----------------------------
try:
    with open("diseases_df.pkl", "rb") as f:
        df = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load diseases_df.pkl: {e}")
    st.stop()

# Convert to lists
disease_symptoms = df["symptoms"].apply(lambda x: str(x)).tolist()
diseases = df["condition"].astype(str).tolist()

# Use recommendations from DataFrame
recommendations_dict = dict(zip(df["condition"], df["recommendations"].astype(str)))

# -----------------------------
# Use pre-trained clustering labels
# -----------------------------
X_agg = tfidf_vectorizer_agg.transform(disease_symptoms).toarray()
df["agg_cluster"] = agg_model.labels_  # استخدم الـ labels_ بدون تدريب

# -----------------------------
# Function to find similar diseases
# -----------------------------
def find_similar_disease_agg(query_text, top_k=5, threshold=0.3):
    query_vec_agg = tfidf_vectorizer_agg.transform([query_text]).toarray()

    # Find nearest cluster
    sims_clusters = cosine_similarity(query_vec_agg, X_agg).flatten()
    best_index = sims_clusters.argmax()
    cluster_id = df.iloc[best_index]["agg_cluster"]

    # Filter by cluster
    cluster_df = df[df["agg_cluster"] == cluster_id].copy()
    cluster_X = X_agg[df["agg_cluster"] == cluster_id]

    sims = cosine_similarity(query_vec_agg, cluster_X).flatten()
    top_indices = sims.argsort()[::-1][:top_k]

    results = cluster_df.iloc[top_indices][["condition", "output_text"]].copy()
    results["agg_similarity"] = sims[top_indices]

    # Filter by threshold
    results = results[results["agg_similarity"] >= threshold]

    # Add rank
    results = results.sort_values(by="agg_similarity", ascending=False).reset_index(drop=True)
    results.insert(0, "rank", results.index + 1)

    # Add recommendations
    results["Recommendation"] = results["condition"].apply(
        lambda d: recommendations_dict.get(d, "No recommendation available")
    )

    return results

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("MediMine Application 🩺")

query_text = st.text_area("Enter your symptoms (separate by commas):")

top_k = st.slider("Number of top results to display", min_value=1, max_value=10, value=5)
threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.3)

if st.button("Predict"):
    if query_text.strip() == "":
        st.warning("Please enter your symptoms first!")
    else:
        results_agg_df = find_similar_disease_agg(query_text, top_k=top_k, threshold=threshold)
        if results_agg_df.empty:
            st.info("No diseases found above the threshold.")
        else:
            st.subheader("Predicted Diseases with Similarity & Recommendations")
         
            st.dataframe(results_agg_df, use_container_width=True)
            
            st.subheader("Top 5 Most Likely Diseases")
            st.table(results_agg_df.head(5))
