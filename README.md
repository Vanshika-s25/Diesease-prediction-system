# Disease Prediction System

A full-stack machine learning application that predicts possible diseases based on user-reported symptoms and vital signs.

## Key Features

- Interactive symptom search and selection
- Real-time BMI calculation and categorization
- Machine learning-based disease prediction
- Confidence scores for predictions
- Interactive data visualizations

## Tech Stack

### Frontend
- React 18 with TypeScript
- Tailwind CSS
- Recharts for visualization
- Lucide React for icons
- Vite for build tooling

### Backend
- Python 3.8+
- scikit-learn for ML
- Pandas and NumPy for data processing
- Streamlit as API layer
- Plotly for advanced charting

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.8+
- npm or yarn

### Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/disease-prediction-system.git
cd disease-prediction-system
```

Install frontend dependencies:
```bash
npm install
```

Install backend dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

Start the frontend:
```bash
npm run dev
```

Start the backend:
```bash
python -m streamlit run app.py
```

Access the app at `http://localhost:5173`

## Project Structure

```
disease-prediction-system/
├── src/            # Frontend source code
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── api/            # Backend (Python + ML model)
│   └── model.py
├── public/         # Static assets
└── package.json    # Frontend dependencies
```

## Feature Breakdown

### Symptom Selection
- Searchable and multi-select symptom input
- Real-time filtering and usability focus

### Vital Monitoring
- Inputs for age, blood pressure, heart rate
- BMI classification:
  - Underweight (<18.5)
  - Normal (18.5–24.9)
  - Overweight (25–29.9)
  - Obese (≥30)

### Disease Prediction
- Random Forest Classifier trained on health data
- Uses symptoms, vitals, and measurements
- Returns potential conditions with confidence levels

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to GitHub (`git push origin feature/YourFeature`)
5. Open a Pull Request

