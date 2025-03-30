import streamlit as st
import pickle
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# Load Model and Vectorizer
with open('plagiarism_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Streamlit UI
st.title("Plagiarism Detection System")
st.write("Enter two sentences to check if they are similar.")

# Input fields
text1 = st.text_area("Enter first sentence:")
text2 = st.text_area("Enter second sentence:")

if st.button("Check Similarity"):
    if text1 and text2:
        # Preprocess text (convert to vectors)
        text1_vec = vectorizer.transform([text1])
        text2_vec = vectorizer.transform([text2])
        cosine_sim = cosine_similarity(text1_vec, text2_vec)[0][0]
        
        # Combine Features
        sample_features = sp.hstack([text1_vec, text2_vec, sp.csr_matrix([cosine_sim])])
        
        # Make Prediction
        prediction = model.predict(sample_features)
        
        # Show Result
        if prediction[0] == 1:
            st.success("The sentences are similar (Plagiarized).")
        else:
            st.warning("The sentences are NOT similar.")
    else:
        st.error("Please enter both sentences!")
