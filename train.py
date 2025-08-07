import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

def clean_data(df):
    df = df.drop(columns=['A_id'], errors='ignore')
    df = df.dropna()
    df = df.astype({'Acidity': 'float64'})
    
    def label(Quality):
        if Quality == "good":
            return 0
        if Quality == "bad":
            return 1
        return None
    
    df['Label'] = df['Quality'].apply(label)
    df = df.drop(columns=['Quality'])
    df = df.astype({'Label': 'int64'})
    return df

def train_and_save_model():
    # Load and clean data
    df = pd.read_csv("data.csv")
    df_clean = clean_data(df.copy())
    
    # Preprocessing
    numerical_features = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']
    
    # Scale features
    robust_scaler = RobustScaler()
    df_clean[numerical_features] = robust_scaler.fit_transform(df_clean[numerical_features])
    
    scaler = StandardScaler()
    df_clean[numerical_features] = scaler.fit_transform(df_clean[numerical_features])
    
    # Prepare data
    X = df_clean.drop(['Label'], axis=1)
    y = df_clean['Label']
    
    # MinMax scaling for final model
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = pd.DataFrame(minmax_scaler.fit_transform(X), columns=X.columns)
    
    # SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Train best SVC model
    param_dist = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 1],
    }
    
    svc = SVC(probability=True)  # Enable probability for confidence scores
    randomized_search = RandomizedSearchCV(svc, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
    randomized_search.fit(X_train, y_train)
    
    best_model = randomized_search.best_estimator_
    
    # Test model
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions, output_dict=True)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(best_model, 'models/apple_quality_model.pkl')
    joblib.dump(minmax_scaler, 'models/scaler.pkl')
    
    # Save model performance metrics
    model_metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'best_params': randomized_search.best_params_
    }
    joblib.dump(model_metrics, 'models/model_metrics.pkl')
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Best parameters: {randomized_search.best_params_}")
    
    return best_model, minmax_scaler, model_metrics

if __name__ == "__main__":
    train_and_save_model()
