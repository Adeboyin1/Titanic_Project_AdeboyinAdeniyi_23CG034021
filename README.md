# Titanic Survival Prediction System

A machine learning-powered web application that predicts whether a passenger would have survived the Titanic disaster based on their characteristics.

## 📋 Project Overview

This project uses a **Random Forest Classifier** to predict survival probability based on five passenger features:
- Passenger Class (Pclass)
- Sex
- Age
- Fare
- Port of Embarkation (Embarked)

## 🛠️ Technology Stack

- **Machine Learning**: scikit-learn (Random Forest Classifier)
- **Web Framework**: Flask
- **Model Persistence**: Joblib
- **Data Processing**: pandas, numpy

## 📁 Project Structure

```
Titanic_Project_yourName_matricNo/
│
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── Titanic_hosted_webGUI_link.txt # Deployment information
│
├── model/
│   ├── model_building.ipynb       # Model development notebook
│   ├── titanic_survival_model.pkl # Trained model
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoder_sex.pkl      # Sex encoder
│   ├── label_encoder_embarked.pkl # Embarked encoder
│   └── feature_names.pkl          # Feature names list
│
└── templates/
    └── index.html                 # Web interface
```

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Titanic_Project_yourName_matricNo.git
cd Titanic_Project_yourName_matricNo
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

Run the Jupyter notebook or execute the model building script:

```bash
jupyter notebook model/model_building.ipynb
```

Or convert to Python script and run:

```bash
jupyter nbconvert --to python model/model_building.ipynb
python model/model_building.py
```

This will generate:
- `titanic_survival_model.pkl`
- `scaler.pkl`
- `label_encoder_sex.pkl`
- `label_encoder_embarked.pkl`
- `feature_names.pkl`

### 5. Run the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## 🌐 Deployment Instructions

### Option 1: Deploy to Render.com

1. Create account on [Render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click "Create Web Service"

### Option 2: Deploy to PythonAnywhere.com

1. Create account on [PythonAnywhere.com](https://www.pythonanywhere.com)
2. Upload files via Files tab
3. Create new web app (Flask)
4. Configure WSGI file to point to your app
5. Install requirements in Bash console:
   ```bash
   pip install --user -r requirements.txt
   ```
6. Reload web app

### Option 3: Deploy to Streamlit Cloud

Create a Streamlit version (`streamlit_app.py`):

```python
import streamlit as st
import joblib
import pandas as pd

st.title("🚢 Titanic Survival Prediction")

# Load model
model = joblib.load('model/titanic_survival_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.number_input("Age", min_value=0, max_value=120, value=29)
fare = st.number_input("Fare ($)", min_value=0.0, value=7.25)
embarked = st.selectbox("Port", ["Cherbourg", "Queenstown", "Southampton"])

if st.button("Predict"):
    # Encode inputs
    sex_encoded = 0 if sex == "Female" else 1
    embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
    embarked_encoded = embarked_map[embarked]
    
    # Create dataframe
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'Fare': [fare],
        'Embarked': [embarked_encoded]
    })
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display result
    if prediction == 1:
        st.success(f"✅ Survived (Probability: {probability[1]*100:.2f}%)")
    else:
        st.error(f"❌ Did Not Survive (Probability: {probability[0]*100:.2f}%)")
```

Deploy to [Streamlit Cloud](https://streamlit.io/cloud)

## 📊 Model Performance

The Random Forest Classifier achieves:
- **Accuracy**: ~80-82%
- **Features Used**: Pclass, Sex, Age, Fare, Embarked

Classification report includes:
- Precision
- Recall
- F1-Score
- Support

## 🎯 Features

- **5 Input Features**: Carefully selected from the Titanic dataset
- **Data Preprocessing**: Handles missing values, encodes categorical variables
- **Feature Scaling**: StandardScaler for optimal model performance
- **Model Persistence**: Saved using Joblib for easy deployment
- **Interactive Web UI**: Clean, responsive design
- **Real-time Predictions**: Instant survival probability calculation

## 📝 Usage Example

**Input:**
- Passenger Class: 3
- Sex: Male
- Age: 22
- Fare: $7.25
- Embarked: Southampton

**Output:**
- Prediction: Did Not Survive
- Survival Probability: 15.23%

## 🔍 Model Details

- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 trees
- **Max Depth**: 10
- **Train/Test Split**: 80/20
- **Stratification**: Applied to maintain class balance

## 📦 File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Flask web application entry point |
| `requirements.txt` | Python package dependencies |
| `model_building.ipynb` | Complete model development pipeline |
| `index.html` | User interface template |
| `*.pkl` files | Serialized model and preprocessing objects |

## 🤝 Contributing

This is an academic project. For questions or improvements, contact the project author.

## 📄 License

This project is created for educational purposes as part of a Machine Learning course.

## 👤 Author

**Your Name**  
Matric Number: Your_Matric_Number  
Institution: Your Institution

## 🙏 Acknowledgments

- Dataset: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- Framework: Flask
- ML Library: scikit-learn