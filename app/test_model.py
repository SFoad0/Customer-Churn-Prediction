import joblib
import pandas as pd

def preprocess_input(data):
    df = pd.DataFrame([data])
    df.drop('customerID', axis=1, inplace=True)
    df.drop('gender', axis=1, inplace=True)
    df = df.reset_index(drop=True)

    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']

    df_services = df[services].apply(lambda col: col.map(lambda x: 1 if x == 'Yes' else 0))
    df['InternetServicesUsed'] = df_services.sum(axis=1)
    
    # binary_columns
    binary_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})

    # InternetService_column
    df['DLS'] = df['InternetService'].apply(lambda x: 1 if x == 'DSL' else 0)
    df['Fiber'] = df['InternetService'].apply(lambda x: 1 if x == 'Fiber optic' else 0)
    df = df.drop(columns=['InternetService'])

    # triple_columns
    triple_col = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for column in triple_col:
        df[f"{column}_y"] = df[column].apply(lambda x: 1 if x == 'Yes' else 0)
        df = df.drop(columns=[column])

    #categorical_columns
    cat_cols = ['Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols)
    df = df.astype({col: int for col in df.select_dtypes('bool').columns})

    model_columns = joblib.load("app/model_columns.pkl")

    df = pd.DataFrame(df, columns=model_columns)
    df = df.fillna(0)

    scaler = joblib.load("app/orginal_scaler.pkl")
    df = scaler.transform(df)


    return df


def predict_customer_churn(customer_data):
    model = joblib.load("app/model.pkl") 
    X = preprocess_input(customer_data)
    probas = model.predict_proba(X)[0]    
    prob_yes = round(float(probas[1]), 2)
    prediction = "Yes" if prob_yes > 0.6 else "No"
    return {"prediction": prediction, "probability_of_yes": prob_yes}

if __name__ == "__main__":
    sample = {
    'customerID': '0001',
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 12,
    'PhoneService': 'Yes',
    'PaperlessBilling': 'No',
    'MonthlyCharges': 70.35,
    'TotalCharges': '845.5',
    'InternetService': 'DSL',
    'Contract': 'Month-to-month',
    'PaymentMethod': 'Electronic check',
    'OnlineSecurity': 'NO',
    'OnlineBackup': 'No',
    'DeviceProtection': 'NO',
    'TechSupport': 'Yes',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'MultipleLines': 'Yes'
    }


    result = predict_customer_churn(sample)
    print("Churn prediction:", result)