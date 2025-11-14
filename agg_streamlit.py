import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------
# Load the saved clustering model and vectorizer safely
# ------------------------------------------------------
try:
    with open("agglomerative_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load agglomerative_model.pkl: {e}")
    st.stop()

try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load vectorizer.pkl: {e}")
    st.stop()

# Load recommendations file
try:
    with open("recommendations.pkl", "rb") as f:
        recommendations = pickle.load(f)
except:
    recommendations = {}
    st.warning("Warning: recommendations.pkl not found.")

# ------------------------------------------------------
# Streamlit User Interface
# ------------------------------------------------------
st.title("Medical Disease Predictor 🩺")

# Input box for symptoms
user_input = st.text_area("Enter your symptoms (separate by commas):")

# ------------------------------------------------------
# When user clicks Predict button
# ------------------------------------------------------
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter your symptoms first!")
        st.stop()

    # Convert patient symptoms to TF-IDF
    user_vec = vectorizer.transform([user_input])

    # ------------------------------------------------------
    # FIX: Handle disease_symptoms from the model safely
    # ------------------------------------------------------
    disease_symptoms = model.get("disease_symptoms", [])

    # If disease_symptoms is a single string → convert to list
    if isinstance(disease_symptoms, str):
        disease_symptoms = [disease_symptoms]

    # If it's a dictionary → take values
    elif isinstance(disease_symptoms, dict):
        disease_symptoms = list(disease_symptoms.values())

    # If it's numpy array / tuple / other → convert to list
    elif not isinstance(disease_symptoms, list):
        try:
            disease_symptoms = list(disease_symptoms)
        except:
            st.error("Error: 'disease_symptoms' is in an unsupported format.")
            st.stop()

    # Convert all entries to strings to avoid TF-IDF errors
    disease_symptoms = [str(s) for s in disease_symptoms]

    # Now transform safely
    disease_vecs = vectorizer.transform(disease_symptoms)

    # ------------------------------------------------------
    # Calculate similarity
    # ------------------------------------------------------
    similarity = cosine_similarity(user_vec, disease_vecs)[0]

    # Create output DataFrame
    df = pd.DataFrame({
        "Disease": model.get("diseases", []),
        "Similarity": similarity
    })

    # Sort diseases by similarity score
    df = df.sort_values(by="Similarity", ascending=False)

    # Add recommendations for each disease
    df["Recommendation"] = df["Disease"].apply(
        lambda d: recommendations.get(d, "No recommendation available")
    )

    # ------------------------------------------------------
    # Display full result table
    # ------------------------------------------------------
    st.subheader("Predicted Diseases with Similarity Scores & Recommendations")
    st.dataframe(df)

    # ------------------------------------------------------
    # Show Top 5 predictions
    # ------------------------------------------------------
    st.subheader("Top 5 Most Likely Diseases")
    st.table(df.head(5))
