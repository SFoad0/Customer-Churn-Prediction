# Note:

**این پروژه هنوز کامل نشده است و فاقد markdown نویسی است و بیشتر برای آشنایی با API بوده است.**

**This project is still under development and lacks full markdown documentation. It mainly serves as an introduction to working .with APIs.**



# Customer Churn Prediction

This project predicts whether a customer will churn (leave the service) based on their usage and account features. The model is trained using the Telco Customer Churn dataset.

## 🚀 Features
- Data Preprocessing and Feature Engineering
- Exploratory Data Analysis (EDA)
- Machine Learning Model Training and Evaluation
- Flask API for model deployment
- Docker-ready setup for containerized deployment

---

## 📁 Project Structure

```
.
│   .gitignore
│   Dockerfile
│   README.md
│   requirements.txt
│
├───app
│   │   app.py                 # Flask API
│   │   preprocess.py          # Preprocessing function for API
│   │   model.pkl              # Trained XGBoost model
│   │   model_columns.pkl      # List of model feature names
│   │   orginal_scaler.pkl     # Scaler used on original data
│   │   resampled_scaler.pkl   # Scaler used on resampled data
│   │   test_model.py          # Local test script for prediction
│
├───data
│       WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│
├───notebooks
│       EDA.ipynb             # Exploratory data analysis
│       modeling.ipynb        # Model training and evaluation
│       PreProcessing.ipynb   # Feature engineering
│
├───src
│       PreProccesing.py      # Preprocessing code used in notebooks
```

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the API
```bash
cd app
python app.py
```

### 4. Test with Postman
Send a POST request to:
```
http://127.0.0.1:5000/predict
```
With JSON body:
```json
{
  "customerID": "00034",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "PaperlessBilling": "No",
  "MonthlyCharges": 70.35,
  "TotalCharges": "845.5",
  "InternetService": "DSL",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "Yes",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "MultipleLines": "Yes"
}
```

---

## 🐳 Docker Deployment (Optional)
You can containerize the app with:

```bash
docker build -t churn-api .
docker run -p 5000:5000 churn-api
```

---

## 🧪 Requirements

- Python 3.10
- Flask
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- joblib
