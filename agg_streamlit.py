import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load the saved model safely
# -----------------------------
try:
    with open("agg_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load agglomerative_model.pkl: {e}")
    st.stop()

# -----------------------------
# Load the vectorizer
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
# Streamlit UI
# -----------------------------
st.title("MediMine Application 🩺")

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

    # -----------------------------
    # Handle disease symptoms and diseases from the model
    # -----------------------------
    # Try dict access first
    try:
        disease_symptoms = model.get("disease_symptoms", [])
        diseases = model.get("diseases", [])
    except AttributeError:
        # If model is object, try attribute access
        disease_symptoms = getattr(model, "disease_symptoms", [])
        diseases = getattr(model, "diseases", [])

    # Ensure disease_symptoms is a list of strings
    if isinstance(disease_symptoms, str):
        disease_symptoms = [disease_symptoms]
    elif isinstance(disease_symptoms, dict):
        disease_symptoms = list(disease_symptoms.values())
    elif not isinstance(disease_symptoms, list):
        try:
            disease_symptoms = list(disease_symptoms)
        except:
            st.error("Error: disease_symptoms format not supported.")
            st.stop()

    disease_symptoms = [str(s) for s in disease_symptoms]

    # Transform disease symptoms using TF-IDF
    disease_vecs = vectorizer.transform(disease_symptoms)

    # -----------------------------
    # Compute cosine similarity
    # -----------------------------
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

    # -----------------------------
    # Display full table
    # -----------------------------
    st.subheader("Predicted Diseases with Similarity & Recommendations")
    st.dataframe(df)

    # -----------------------------
    # Display top 5 predictions
    # -----------------------------
    st.subheader("Top 5 Most Likely Diseases")
    st.table(df.head(5))

