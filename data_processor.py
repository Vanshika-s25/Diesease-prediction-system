import pandas as pd
import numpy as np
import os

# Function to generate synthetic dataset for disease prediction
def generate_synthetic_dataset():
    # Define diseases and their common symptoms
    disease_symptoms = {
        'Common Cold': ['cough', 'runny_nose', 'sore_throat', 'sneezing', 'headache', 'fatigue'],
        'Influenza': ['fever', 'cough', 'fatigue', 'body_aches', 'headache', 'chills'],
        'Pneumonia': ['cough', 'fever', 'shortness_of_breath', 'chest_pain', 'fatigue', 'rapid_breathing'],
        'Tuberculosis': ['cough', 'fever', 'weight_loss', 'night_sweats', 'fatigue', 'chest_pain'],
        'Asthma': ['shortness_of_breath', 'wheezing', 'cough', 'chest_tightness'],
        'Bronchitis': ['cough', 'mucus_production', 'shortness_of_breath', 'chest_discomfort', 'fatigue'],
        'Sinusitis': ['nasal_congestion', 'facial_pain', 'headache', 'cough', 'fatigue'],
        'Gastroenteritis': ['nausea', 'vomiting', 'diarrhea', 'abdominal_cramps', 'fever'],
        'Appendicitis': ['abdominal_pain', 'nausea', 'vomiting', 'fever', 'loss_of_appetite'],
        'Diabetes': ['frequent_urination', 'excessive_thirst', 'weight_loss', 'fatigue', 'blurred_vision'],
        'Hypertension': ['headache', 'shortness_of_breath', 'dizziness', 'chest_pain', 'nosebleeds'],
        'Heart Attack': ['chest_pain', 'shortness_of_breath', 'pain_in_arms', 'dizziness', 'cold_sweat'],
        'Stroke': ['numbness', 'confusion', 'trouble_speaking', 'dizziness', 'severe_headache'],
        'Migraine': ['severe_headache', 'nausea', 'vomiting', 'sensitivity_to_light', 'vision_changes'],
        'Arthritis': ['joint_pain', 'stiffness', 'swelling', 'reduced_mobility', 'redness'],
        'Osteoporosis': ['back_pain', 'loss_of_height', 'stooped_posture', 'bone_fractures'],
        'Anemia': ['fatigue', 'weakness', 'pale_skin', 'shortness_of_breath', 'dizziness'],
        'Hyperthyroidism': ['weight_loss', 'increased_appetite', 'anxiety', 'sweating', 'tremor'],
        'Hypothyroidism': ['fatigue', 'weight_gain', 'cold_intolerance', 'dry_skin', 'hair_loss'],
        'Depression': ['persistent_sadness', 'loss_of_interest', 'sleep_problems', 'fatigue', 'feelings_of_guilt'],
        'Anxiety Disorder': ['excessive_worry', 'restlessness', 'fatigue', 'difficulty_concentrating', 'sleep_problems'],
        'Urinary Tract Infection': ['burning_urination', 'frequent_urination', 'cloudy_urine', 'strong_odor', 'pelvic_pain'],
        'Kidney Stones': ['severe_pain', 'nausea', 'vomiting', 'blood_in_urine', 'frequent_urination'],
        'Malaria': ['fever', 'chills', 'headache', 'nausea', 'vomiting', 'muscle_pain'],
        'Dengue Fever': ['high_fever', 'severe_headache', 'joint_pain', 'rash', 'pain_behind_eyes'],
    }
    
    # Create empty lists for data
    data = []
    
    # Generate multiple records for each disease with different symptom combinations
    for disease, symptoms in disease_symptoms.items():
        n_samples = 30  # Number of samples per disease
        for _ in range(n_samples):
            # Choose a random number of symptoms (at least 3, up to all)
            n_symptoms = np.random.randint(3, len(symptoms) + 1)
            selected_symptoms = np.random.choice(symptoms, size=n_symptoms, replace=False)
            
            # Create a symptoms string
            symptoms_str = ' '.join(selected_symptoms)
            
            # Add the record
            data.append({
                'disease': disease,
                'symptoms_list': symptoms_str
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save the dataset
    df.to_csv('data/disease_symptom_dataset.csv', index=False)
    
    return df

# Function to load dataset
def load_dataset():
    # Check if dataset exists, if not generate it
    if not os.path.exists('data/disease_symptom_dataset.csv'):
        return generate_synthetic_dataset()
    
    # Load the dataset
    df = pd.read_csv('data/disease_symptom_dataset.csv')
    return df

# Function to get list of all symptoms
def get_symptoms_list():
    df = load_dataset()
    
    # Get all symptoms from the dataset
    all_symptoms = set()
    for symptoms_str in df['symptoms_list']:
        symptoms = symptoms_str.split()
        all_symptoms.update(symptoms)
    
    return sorted(list(all_symptoms))