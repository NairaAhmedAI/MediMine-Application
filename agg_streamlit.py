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
# Load recommendations (optional)
# -----------------------------
try:
    with open("recommendations.pkl", "rb") as f:
        recommendations = pickle.load(f)
except:
    recommendations = {}
    st.warning("Recommendations file not found. Proceeding without it.")

# -----------------------------
# Load Disease DataFrame
# -----------------------------
try:
    Disease_df = pd.read_csv("diseases_df.csv") 
except Exception as e:
    st.error(f"Failed to load Disease_df: {e}")
    st.stop()

# Convert to lists
disease_symptoms = Disease_df["Symptoms"].astype(str).tolist()
diseases = Disease_df["Disease"].astype(str).tolist()

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

    # Add recommendations if available
    df["Recommendation"] = df["Disease"].apply(
        lambda d: recommendations.get(d, "No recommendation available")
    )

    # Display full table
    st.subheader("Predicted Diseases with Similarity & Recommendations")
    st.dataframe(df)

    # Display top 5 predictions
    st.subheader("Top 5 Most Likely Diseases")
    st.table(df.head(5))


