from pandas import get_dummies 
from pandas import read_csv
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTENC




def load_data(path):
    df = read_csv(path)
    return df


def split_X_y(dataframe):
    df = dataframe.copy()

    df.drop('customerID', axis=1, inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)
    df.drop('gender', axis=1, inplace=True)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    return X, y



def encode_features(dataframe):

    df = dataframe.copy()
    
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
    df = get_dummies(df, columns=cat_cols)
    df = df.astype({col: int for col in df.select_dtypes('bool').columns})

    
    return df



def undersampling_data(dataframe_X, series_y):

    X_train_resampled = dataframe_X.copy()
    y_train_resampled = series_y.copy()

    X_train_resampled = encode_features(X_train_resampled)
    y_train_resampled = y_train_resampled.map({'No': 0, 'Yes': 1})

    enn = EditedNearestNeighbours(kind_sel = 'mode', n_neighbors=3)
    X_enn, y_enn = enn.fit_resample(dataframe_X, series_y)
    kept_indices = enn.sample_indices_
    X_train_resampled = X_train_resampled.iloc[kept_indices].reset_index(drop=True)
    y_train_resampled = y_train_resampled.iloc[kept_indices].reset_index(drop=True)
    
    print("Before ENN:")
    print(series_y.map({0: 'No', 1: 'Yes'}).value_counts())

    print("After ENN:")
    print(y_enn.map({0: 'No', 1: 'Yes'}).value_counts())

    return X_train_resampled, y_train_resampled



def oversampling_data(dataframe_X, series_y):

    X_train_resampled = dataframe_X.copy()
    y_train_resampled = series_y.copy()

    categorical_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                       'MultipleLines', 'PaperlessBilling', 'InternetService',
                       'OnlineSecurity','OnlineBackup', 'DeviceProtection',
                       'TechSupport','StreamingTV', 'StreamingMovies',
                       'Contract', 'PaymentMethod']
    for col in categorical_columns:
        X_train_resampled[col] = X_train_resampled[col].astype('category')

    cat_idx = [X_train_resampled.columns.get_loc(col) for col in categorical_columns]
    smote_nc = SMOTENC(categorical_features=cat_idx, random_state=42)
    X_resampled, y_resampled = smote_nc.fit_resample(X_train_resampled, y_train_resampled)

    def remove_logical_conflicts(X, y, dependency_col, service_col, service_values, invalid_vals_by_service):
        for service_val in service_values:
            mask_service = X[service_col] == service_val
            invalid_vals = invalid_vals_by_service[service_val]
            mask_conflict = mask_service & X[dependency_col].apply(
                lambda row: any(val in invalid_vals for val in row), axis=1)
            X = X[~mask_conflict].reset_index(drop=True)
            y = y[~mask_conflict].reset_index(drop=True)
        return X, y

    internetService_dependent_col = ['StreamingMovies', 'StreamingTV', 'TechSupport', 'DeviceProtection', 'OnlineBackup', 'OnlineSecurity']
    phoneService_dependent_col = ['MultipleLines']

    internet_invalids = {
        'No': ['Yes', 'No'],
        'DSL': ['No internet service'],
        'Fiber optic': ['No internet service']
    }
    phone_invalids = {
        'No': ['Yes', 'No'],
        'Yes': ['No phone service']
    }

    X_resampled, y_resampled = remove_logical_conflicts(
        X_resampled, y_resampled, internetService_dependent_col, 'InternetService',
        ['No', 'DSL', 'Fiber optic'], internet_invalids
    )

    X_resampled, y_resampled = remove_logical_conflicts(
        X_resampled, y_resampled, phoneService_dependent_col, 'PhoneService',
        ['No', 'Yes'], phone_invalids
    )

    print("Before SMOTE:")
    print(series_y.value_counts())
    print("After SMOTE:")
    print(y_resampled.value_counts())

    return X_resampled, y_resampled
    


def encode_cat_num(Xtrain, Xtest, ytrain, ytest):
       
    X_train = encode_features(Xtrain)
    X_test = encode_features(Xtest)
    y_train = ytrain.map({'No': 0, 'Yes': 1})
    y_test = ytest.map({'No': 0, 'Yes': 1})

    return X_train, X_test, y_train, y_test