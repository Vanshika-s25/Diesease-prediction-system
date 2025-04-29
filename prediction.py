import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from data_processor import load_dataset, get_symptoms_list

# Function to create and save model (run only once)
def train_and_save_model():
    # Load and prepare data
    df = load_dataset()
    
    # Create feature matrix
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['symptoms_list'])
    y = df['disease']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model and vectorizer
    os.makedirs('models', exist_ok=True)
    with open('models/disease_prediction_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return model, vectorizer

# Function to load model
def load_model():
    # Check if model exists, if not train a new one
    if not os.path.exists('models/disease_prediction_model.pkl'):
        return train_and_save_model()
    
    # Load model and vectorizer
    with open('models/disease_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

# Function to make prediction
def predict_disease(symptoms, model, vectorizer):
    # Convert symptoms to string format expected by vectorizer
    symptoms_str = ' '.join(symptoms)
    
    # Transform symptoms
    symptoms_vector = vectorizer.transform([symptoms_str])
    
    # Make prediction
    prediction_probs = model.predict_proba(symptoms_vector)[0]
    disease_indices = prediction_probs.argsort()[-5:][::-1]  # Get indices of top 5 diseases
    
    top_diseases = []
    for idx in disease_indices:
        disease = model.classes_[idx]
        probability = prediction_probs[idx] * 100
        top_diseases.append((disease, probability))
    
    # Get top prediction
    predicted_disease = top_diseases[0][0]
    probability = top_diseases[0][1]
    
    # Save last prediction in session state
    import streamlit as st
    st.session_state.last_prediction = predicted_disease
    st.session_state.last_symptoms = symptoms
    
    return predicted_disease, probability, top_diseases