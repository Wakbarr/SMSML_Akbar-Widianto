import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mlflow
import mlflow.sklearn
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

def load_and_prepare_data():
    """
    Load preprocessed data and prepare for modeling
    """
    try:
        # Load preprocessed data
        df = pd.read_csv('membangun_model/folder_dataset/penguins_processed.csv')
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Define features and target
        feature_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                          'body_mass_g', 'island_encoded', 'sex_encoded']
        
        X = df[feature_columns]
        y = df['species_encoded']
        
        logging.info(f"Features shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")
        logging.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def train_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train model with MLflow tracking
    """
    with mlflow.start_run(run_name=model_name):
        # Enable autolog
        mlflow.sklearn.autolog()
        
        logging.info(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log additional metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("model_type", model_name)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"{model_name}_penguins"
        )
        
        logging.info(f"{model_name} - Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, accuracy

def main():
    """
    Main function to execute the modeling pipeline
    """
    # Set MLflow tracking URI (local)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Palmer_Penguins_Classification")
    
    logging.info("Starting Palmer Penguins Classification Model Training")
    
    try:
        # Load and prepare data
        X, y = load_and_prepare_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logging.info(f"Train set shape: {X_train.shape}")
        logging.info(f"Test set shape: {X_test.shape}")
        
        # Define models to train
        models = {
            "Random_Forest": RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10
            ),
            "Logistic_Regression": LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            "SVM": SVC(
                random_state=42,
                kernel='rbf',
                C=1.0
            )
        }
        
        # Train models and track with MLflow
        results = {}
        best_accuracy = 0
        best_model_name = ""
        
        for model_name, model in models.items():
            trained_model, accuracy = train_model(
                model, model_name, X_train, X_test, y_train, y_test
            )
            results[model_name] = {
                'model': trained_model,
                'accuracy': accuracy
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
        
        # Log best model information
        logging.info(f"\nBest Model: {best_model_name}")
        logging.info(f"Best Accuracy: {best_accuracy:.4f}")
        
        # Print results summary
        print("\n" + "="*50)
        print("MODEL TRAINING RESULTS SUMMARY")
        print("="*50)
        
        for model_name, result in results.items():
            print(f"{model_name}: {result['accuracy']:.4f}")
        
        print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        print("="*50)
        
        logging.info("Model training completed successfully!")
        logging.info("Check MLflow UI at: http://localhost:5000")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()