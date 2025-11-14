import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load TF-IDF vectorizer
# -----------------------------
try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load vectorizer.pkl: {e}")
    st.stop()

# -----------------------------
# Load Disease DataFrame from pickle
# -----------------------------
try:
    with open("diseases_df.pkl", "rb") as f:
        Disease_df = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load Disease_df.pkl: {e}")
    st.stop()

# Convert to lists
disease_symptoms = Disease_df["symptoms"].apply(lambda x: str(x)).tolist()
diseases = Disease_df["condition"].astype(str).tolist()

# Use recommendations from DataFrame if exists
recommendations_dict = dict(zip(Disease_df["condition"], Disease_df["recommendations"].astype(str)))

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("MediMine Application🩺")

# User input for symptoms
user_input = st.text_area("Enter your symptoms (separate by commas):")

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter your symptoms first!")
        st.stop()

    # Convert user symptoms to TF-IDF
    user_vec = vectorizer.transform([user_input])

    # Transform disease symptoms
    disease_vecs = vectorizer.transform(disease_symptoms)

    # Compute cosine similarity
    similarity = cosine_similarity(user_vec, disease_vecs)[0]

    # Create a DataFrame with results
    df = pd.DataFrame({
        "Disease": diseases,
        "Similarity": similarity
    })

    # Sort by similarity descending
    df = df.sort_values(by="Similarity", ascending=False)

    # Add recommendations
    df["Recommendation"] = df["Disease"].apply(
        lambda d: recommendations_dict.get(d, "No recommendation available")
    )

    # Display full table
    st.subheader("Predicted Diseases with Similarity & Recommendations")
    st.dataframe(df)

    # Display top 5 predictions
    st.subheader("Top 5 Most Likely Diseases")
    st.table(df.head(5))





