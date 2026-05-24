import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def prepare_data():
    # Load dataset
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Fix TotalCharges column (it has spaces, convert to number)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Drop CustomerID (not useful for prediction)
    df.drop('customerID', axis=1, inplace=True)
    
    # Convert target column Yes/No to 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Convert all text columns to numbers
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Train Test Split — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Preprocessing Done!")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    prepare_data()