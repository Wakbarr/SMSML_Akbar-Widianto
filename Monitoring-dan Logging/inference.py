import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
import time
import requests
from prometheus_exporter import MLMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PenguinPredictor:
    def __init__(self, model_name="Random_Forest_penguins", model_version="latest"):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.monitor = MLMonitor()
        self.species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
        
    def load_model(self):
        """Load model from MLflow"""
        try:
            mlflow.set_tracking_uri("file:./mlruns")
            model_uri = f"models:/{self.model_name}/{self.model_version}"
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model {self.model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_single(self, features):
        """Make single prediction"""
        start_time = time.time()
        
        try:
            # Ensure features is a DataFrame with correct column names
            if isinstance(features, dict):
                features_df = pd.DataFrame([features])
            elif isinstance(features, list):
                feature_names = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                               'body_mass_g', 'island_encoded', 'sex_encoded']
                features_df = pd.DataFrame([features], columns=feature_names)
            else:
                features_df = features
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            probability = self.model.predict_proba(features_df)[0]
            
            # Convert to species name
            species = self.species_mapping[prediction]
            confidence = max(probability)
            
            # Record metrics
            duration = time.time() - start_time
            self.monitor.record_request('POST', '/predict', 200, duration)
            self.monitor.record_prediction(self.model_name, species)
            
            result = {
                'predicted_species': species,
                'confidence': float(confidence),
                'probabilities': {
                    'Adelie': float(probability[0]),
                    'Chinstrap': float(probability[1]),
                    'Gentoo': float(probability[2])
                },
                'prediction_time': duration
            }
            
            logger.info(f"Prediction: {species} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_request('POST', '/predict', 500, duration)
            logger.error(f"Prediction failed: {e}")
            return {'error': str(e)}
    
    def predict_batch(self, features_list):
        """Make batch predictions"""
        results = []
        for features in features_list:
            result = self.predict_single(features)
            results.append(result)
        return results
    
    def start_monitoring(self):
        """Start monitoring"""
        self.monitor.start_monitoring()

# Example usage and testing
def test_predictions():
    """Test the predictor with sample data"""
    predictor = PenguinPredictor()
    
    # Start monitoring
    predictor.start_monitoring()
    
    # Load model
    if not predictor.load_model():
        logger.error("Cannot proceed without model")
        return
    
    # Sample penguin data for testing
    test_samples = [
        {
            'bill_length_mm': 39.1,
            'bill_depth_mm': 18.7,
            'flipper_length_mm': 181.0,
            'body_mass_g': 3750.0,
            'island_encoded': 0,  # Torgersen
            'sex_encoded': 1      # Male
        },
        {
            'bill_length_mm': 46.5,
            'bill_depth_mm': 17.9,
            'flipper_length_mm': 192.0,
            'body_mass_g': 3500.0,
            'island_encoded': 1,  # Biscoe
            'sex_encoded': 0      # Female
        },
        {
            'bill_length_mm': 50.7,
            'bill_depth_mm': 19.7,
            'flipper_length_mm': 203.0,
            'body_mass_g': 4725.0,
            'island_encoded': 1,  # Biscoe
            'sex_encoded': 1      # Male
        }
    ]
    
    logger.info("Starting prediction tests...")
    
    try:
        # Test individual predictions
        for i, sample in enumerate(test_samples):
            logger.info(f"\nTest sample {i+1}:")
            logger.info(f"Input: {sample}")
            
            result = predictor.predict_single(sample)
            logger.info(f"Result: {result}")
            
            time.sleep(2)  # Small delay between predictions
        
        # Test batch prediction
        logger.info("\nTesting batch prediction...")
        batch_results = predictor.predict_batch(test_samples)
        logger.info(f"Batch results: {len(batch_results)} predictions completed")
        
        # Simulate continuous predictions for monitoring
        logger.info("\nSimulating continuous predictions for monitoring...")
        for i in range(10):
            import random
            sample = random.choice(test_samples)
            
            # Add some noise to the sample
            noisy_sample = sample.copy()
            noisy_sample['bill_length_mm'] += random.uniform(-2, 2)
            noisy_sample['bill_depth_mm'] += random.uniform(-1, 1)
            
            result = predictor.predict_single(noisy_sample)
            logger.info(f"Simulation {i+1}: {result['predicted_species']} (confidence: {result['confidence']:.3f})")
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        logger.info("Stopping predictions...")
    except Exception as e:
        logger.error(f"Error during testing: {e}")

if __name__ == "__main__":
    test_predictions()