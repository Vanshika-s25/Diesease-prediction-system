# MediPredict - Disease Prediction System

A machine learning-based disease prediction system with a Streamlit interface.

## Overview

MediPredict is an educational tool that demonstrates the application of machine learning algorithms in predicting potential diseases based on patient symptoms. The system uses a Random Forest classifier trained on a synthetic dataset of disease-symptom relationships.

## Features

- User-friendly interface for inputting patient symptoms
- Machine learning model for disease prediction
- Visualization of prediction results with probability scores
- Information display about predicted diseases
- Simple patient data management system
- Responsive layout that works across different devices

## Technical Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework for the UI
- **Scikit-learn**: Machine learning library for the prediction model
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/medipredict.git
   cd medipredict
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

5. Open your browser and navigate to http://localhost:8501

## How It Works

1. The system uses a machine learning model trained on a dataset of disease-symptom relationships
2. Users input symptoms through a user-friendly interface
3. The selected symptoms are processed and fed into the prediction model
4. The model returns the most likely diseases along with their probability scores
5. Results are displayed in an easy-to-understand format

## Disclaimer

This system is designed for educational purposes only and should not be used for actual medical diagnosis. Always consult with healthcare professionals for proper medical advice and treatment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.