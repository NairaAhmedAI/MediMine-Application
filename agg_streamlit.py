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
st.title("Medical Disease Predictor 🩺")

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
    # Handle disease symptoms and diseases from the model safely
    # -----------------------------
    # Try dict access first
    try:
        disease_symptoms = model.get("disease_symptoms", None)
        diseases = model.get("diseases", None)
    except AttributeError:
        # If model is object, try attribute access
        disease_symptoms = getattr(model, "disease_symptoms", None)
        diseases = getattr(model, "diseases", None)

    # -----------------------------
    # FIX: Ensure disease_symptoms is a valid list of strings
    # -----------------------------
    if disease_symptoms is None:
        st.error("Error: disease_symptoms not found in the model.")
        st.stop()

    # Convert to list if it's not already
    if not isinstance(disease_symptoms, list):
        try:
            disease_symptoms = list(disease_symptoms)
        except Exception as e:
            st.error(f"Error: disease_symptoms cannot be converted to list. Details: {e}")
            st.stop()

    # Remove None or empty strings and strip spaces
    disease_symptoms = [str(s).strip() for s in disease_symptoms if s is not None and str(s).strip() != ""]

    # Stop if the list is empty
    if len(disease_symptoms) == 0:
        st.error("Error: disease_symptoms is empty after cleaning. Cannot transform.")
        st.stop()

    # Now safe to transform with the TF-IDF vectorizer
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

