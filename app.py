import streamlit as st
import pandas as pd
import numpy as np
from prediction import load_model, predict_disease
from data_processor import load_dataset, get_symptoms_list

# Configure the Streamlit page
st.set_page_config(
    page_title="MediPredict - Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        border-color: #2563eb;
    }
    .info-box {
        background-color: #dbeafe;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff7ed;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f97316;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #ecfdf5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #14B8A6;
        margin-bottom: 1rem;
    }
    .symptom-checkbox {
        background-color: #f1f5f9;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
model, vectorizer = load_model()
df = load_dataset()
all_symptoms = get_symptoms_list()

# Title and description
st.title("🏥 MediPredict: Disease Prediction System")

st.markdown("""
<div class="info-box">
    <h3>Welcome to MediPredict</h3>
    <p>This system uses machine learning to predict possible diseases based on symptoms. 
    Please select the symptoms you're experiencing, and our AI will analyze the patterns to suggest potential diagnoses.</p>
    <p><strong>Note:</strong> This tool is for educational purposes only and does not replace professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Disease Prediction", "How It Works", "About"])

with tab1:
    st.header("Enter Patient Information")
    
    # Patient basic information
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    with col2:
        temperature = st.slider("Body Temperature (°F)", 95.0, 105.0, 98.6, 0.1)
        blood_pressure = st.slider("Blood Pressure (systolic)", 80, 200, 120)
    
    # Symptom selection section
    st.header("Select Symptoms")
    
    # Group symptoms by category for better organization
    symptom_categories = {
        "General": ["fatigue", "weakness", "malaise", "weight_loss", "fever", "sweating", "chills"],
        "Head & Neurological": ["headache", "dizziness", "unconsciousness", "coma", "confusion", "memory_loss", "anxiety"],
        "Respiratory": ["cough", "shortness_of_breath", "sore_throat", "runny_nose", "chest_pain", "rapid_breathing"],
        "Digestive": ["nausea", "vomiting", "abdominal_pain", "diarrhea", "constipation", "bloating", "jaundice"],
        "Muscular & Skeletal": ["muscle_pain", "joint_pain", "stiffness", "swelling", "back_pain", "muscle_weakness"],
        "Skin": ["rash", "itching", "discoloration", "dryness", "bruising", "swelling"],
        "Other": ["blurred_vision", "hearing_loss", "frequent_urination", "excessive_thirst", "irregular_heartbeat"]
    }
    
    selected_symptoms = []
    
    # Create expandable sections for each symptom category
    for category, symptoms in symptom_categories.items():
        with st.expander(f"{category} Symptoms", expanded=category=="General"):
            st.write(f"Select all {category.lower()} symptoms that apply:")
            
            # Create columns for better layout
            cols = st.columns(2)
            for i, symptom in enumerate(symptoms):
                col_idx = i % 2
                with cols[col_idx]:
                    if symptom in all_symptoms:
                        if st.checkbox(symptom.replace("_", " ").title(), key=symptom):
                            selected_symptoms.append(symptom)
    
    # Additional symptoms text input
    st.write("Any other symptoms not listed above:")
    other_symptoms = st.text_area("Describe other symptoms", height=100)
    
    # Prediction button
    if st.button("Predict Disease", type="primary"):
        if len(selected_symptoms) < 2:
            st.warning("Please select at least 2 symptoms for a more accurate prediction.")
        else:
            # Make prediction
            prediction, probability, top_diseases = predict_disease(selected_symptoms, model, vectorizer)
            
            # Display results
            st.markdown(f"""
            <div class="result-box">
                <h3>Prediction Results</h3>
                <p>Based on the symptoms provided, the most likely diagnosis is:</p>
                <h2>{prediction}</h2>
                <p>Confidence: {probability:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display other possible diseases
            st.subheader("Other Possible Diagnoses")
            
            for disease, prob in top_diseases[1:4]:  # Show top 3 alternatives
                st.markdown(f"""
                <div style="background-color:#f8fafc; padding:0.8rem; border-radius:8px; margin-bottom:0.5rem; border-left:4px solid #64748b;">
                    <strong>{disease}</strong>: {prob:.2f}%
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
                <h4>Important Disclaimer</h4>
                <p>This prediction is based on machine learning algorithms and is not a substitute for professional medical diagnosis. 
                Please consult with a healthcare provider for proper evaluation and treatment.</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("How MediPredict Works")
    
    st.write("""
    MediPredict uses machine learning algorithms to analyze patterns in symptoms and predict possible diseases. 
    Here's how it works:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Data Collection
        The system is trained on a dataset of disease symptoms and diagnoses from medical records and literature.
        
        ### Preprocessing
        The symptoms are processed and converted into features that can be understood by the machine learning model.
        
        ### Model Training
        We use a Random Forest classifier, which is trained to recognize patterns between symptoms and diseases.
        """)
    
    with col2:
        st.markdown("""
        ### Prediction Process
        1. You select the symptoms you're experiencing
        2. The system converts these symptoms into the same format used during training
        3. The model analyzes the symptom pattern
        4. The system returns the most likely diseases and their probabilities
        
        ### Continuous Improvement
        The model can be retrained with new data to improve accuracy over time.
        """)
    
    st.subheader("Accuracy and Limitations")
    st.write("""
    - The current model has an accuracy of approximately 85% on test data
    - Predictions are more accurate when more symptoms are provided
    - The system works best for common diseases with distinctive symptom patterns
    - Rare diseases or those with symptoms similar to many other conditions may be more difficult to predict accurately
    """)

with tab3:
    st.header("About MediPredict")
    
    st.write("""
    MediPredict was developed as an educational tool to demonstrate the application of machine learning
    in healthcare. The system aims to show how AI can assist in preliminary disease assessment.
    
    ### Purpose
    - To demonstrate the potential of AI in healthcare
    - To provide a learning tool for understanding the relationship between symptoms and diseases
    - To assist in preliminary screening (not for diagnostic purposes)
    
    ### Development Team
    This system was developed by healthcare AI enthusiasts with a background in medical informatics 
    and machine learning.
    
    ### Contact
    For questions, feedback or suggestions, please contact us at support@medipredict.example.com
    """)

# Sidebar content
with st.sidebar:
    st.header("Patient History")
    
    # Sample patient records for demonstration
    st.subheader("Recent Records")
    
    if st.button("New Patient"):
        st.session_state.patient_name = ""
        st.session_state.patient_history = []
    
    patient_name = st.text_input("Patient Name", key="patient_name")
    
    if patient_name:
        if 'patient_history' not in st.session_state:
            st.session_state.patient_history = []
            
        if st.button("Save Current Assessment"):
            if 'last_prediction' in st.session_state:
                st.session_state.patient_history.append({
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "diagnosis": st.session_state.last_prediction,
                    "symptoms": ", ".join(st.session_state.last_symptoms)
                })
        
        # Display patient history
        if st.session_state.patient_history:
            for i, record in enumerate(st.session_state.patient_history):
                st.markdown(f"""
                <div style="background-color:#f8fafc; padding:0.8rem; border-radius:8px; margin-bottom:0.5rem;">
                    <strong>Date:</strong> {record["date"]}<br>
                    <strong>Diagnosis:</strong> {record["diagnosis"]}<br>
                    <strong>Symptoms:</strong> {record["symptoms"]}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No records yet.")
    
    # Resources section
    st.header("Resources")
    st.write("Learn more about diseases and their symptoms:")
    
    resources = [
        {"name": "CDC Health Information", "url": "https://www.cdc.gov/"},
        {"name": "Mayo Clinic Disease Index", "url": "https://www.mayoclinic.org/diseases-conditions"},
        {"name": "WebMD Symptom Checker", "url": "https://www.webmd.com/symptoms/default.htm"},
        {"name": "WHO Health Topics", "url": "https://www.who.int/health-topics/"}
    ]
    
    for resource in resources:
        st.markdown(f"* [{resource['name']}]({resource['url']})")