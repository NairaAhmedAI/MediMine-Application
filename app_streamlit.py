import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load the saved model and vectorizer
# -----------------------------
with open("agglomerative_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load saved recommendations
with open("recommendations.pkl", "rb") as f:
    recommendations = pickle.load(f)

# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("Medical Disease Predictor 🩺")

# Text input for user symptoms
user_input = st.text_area("Enter your symptoms (separate by commas):")

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter your symptoms first!")

    else:
        # Convert patient symptoms to TF-IDF
        user_vec = vectorizer.transform([user_input])

        # -----------------------------
        # Handle disease symptoms safely
        # This fixes the TypeError on Streamlit Cloud
        # -----------------------------
        disease_symptoms = model.get("disease_symptoms", [])

        # If symptoms is a single string → convert to list
        if isinstance(disease_symptoms, str):
            disease_symptoms = [disease_symptoms]

        # If symptoms are in a dictionary → use values
        elif isinstance(disease_symptoms, dict):
            disease_symptoms = list(disease_symptoms.values())

        # If not a list → convert to list (e.g. numpy array)
        elif not isinstance(disease_symptoms, list):
            try:
                disease_symptoms = list(disease_symptoms)
            except:
                st.error("Error: disease symptoms are not in a valid format.")
                st.stop()

        # Ensure every symptom is a string
        disease_symptoms = [str(s) for s in disease_symptoms]

        # Transform disease symptoms using TF-IDF
        disease_vecs = vectorizer.transform(disease_symptoms)

        # -----------------------------
        # Compute cosine similarity
        # -----------------------------
        similarity = cosine_similarity(user_vec, disease_vecs)[0]

        # Create DataFrame with results
        df = pd.DataFrame({
            "Disease": model["diseases"],
            "Similarity": similarity
        })

        # Sort diseases by similarity (highest first)
        df = df.sort_values(by="Similarity", ascending=False)

        # Add medical recommendations
        df["Recommendation"] = df["Disease"].apply(
            lambda d: recommendations.get(d, "No recommendation available")
        )

        # -----------------------------
        # Display full prediction table
        # -----------------------------
        st.subheader("Predicted Diseases with Similarity & Recommendations")
        st.dataframe(df)

        # -----------------------------
        # Display Top 5
        # -----------------------------
        st.subheader("Top 5 Possible Diseases")
        st.table(df.head(5))
