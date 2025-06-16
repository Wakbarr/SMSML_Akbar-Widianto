from flask import Flask, request, jsonify
import pandas as pd
import mlflow
import mlflow.sklearn
import logging
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_exporter import MLMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize monitor
monitor = MLMonitor()
monitor.start_monitoring()

# Global variables
model = None
species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

def load_model():
    """Load the best model from MLflow"""
    global model
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        # Try to load the Random Forest model (usually performs best)
        model_uri = "models:/Random_Forest_penguins/latest"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully")
        
        # Update model accuracy metric (you can get this from your training results)
        monitor.update_model_accuracy(0.92)  # Example accuracy
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    start_time = time.time()
    
    status = "healthy" if model is not None else "unhealthy"
    response = {"status": status, "timestamp": time.time()}
    
    duration = time.time() - start_time
    status_code = 200 if status == "healthy" else 503
    monitor.record_request('GET', '/health', status_code, duration)
    
    return jsonify(response), status_code

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    start_time = time.time()
    
    try:
        if model is None:
            raise Exception("Model not loaded")
        
        # Get input data
        data = request.get_json()
        
        # Validate input
        required_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                           'body_mass_g', 'island_encoded', 'sex_encoded']
        
        if not all(feature in data for feature in required_features):
            raise ValueError(f"Missing required features: {required_features}")
        
        # Create DataFrame
        features_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        # Convert to species name
        species = species_mapping[prediction]
        confidence = max(probabilities)
        
        # Record metrics
        duration = time.time() - start_time
        monitor.record_request('POST', '/predict', 200, duration)
        monitor.record_prediction('Random_Forest', species)
        
        response = {
            'predicted_species': species,
            'confidence': float(confidence),
            'probabilities': {
                'Adelie': float(probabilities[0]),
                'Chinstrap': float(probabilities[1]),
                'Gentoo': float(probabilities[2])
            },
            'prediction_time': duration
        }
        
        logger.info(f"Prediction: {species} (confidence: {confidence:.3f})")
        return jsonify(response), 200
        
    except Exception as e:
        duration = time.time() - start_time
        monitor.record_request('POST', '/predict', 500, duration)
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    start_time = time.time()
    
    try:
        if model is None:
            raise Exception("Model not loaded")
        
        # Get input data
        data = request.get_json()
        
        if 'samples' not in data:
            raise ValueError("Request must contain 'samples' array")
        
        results = []
        for sample in data['samples']:
            # Validate each sample
            required_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                               'body_mass_g', 'island_encoded', 'sex_encoded']
            
            if not all(feature in sample for feature in required_features):
                results.append({'error': f"Missing required features: {required_features}"})
                continue
            
            # Create DataFrame
            features_df = pd.DataFrame([sample])
            
            # Make prediction
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            
            # Convert to species name
            species = species_mapping[prediction]
            confidence = max(probabilities)
            
            # Record prediction metric
            monitor.record_prediction('Random_Forest', species)
            
            results.append({
                'predicted_species': species,
                'confidence': float(confidence),
                'probabilities': {
                    'Adelie': float(probabilities[0]),
                    'Chinstrap': float(probabilities[1]),
                    'Gentoo': float(probabilities[2])
                }
            })
        
        # Record metrics
        duration = time.time() - start_time
        monitor.record_request('POST', '/predict/batch', 200, duration)
        
        response = {
            'results': results,
            'total_predictions': len(results),
            'processing_time': duration
        }
        
        logger.info(f"Batch prediction: {len(results)} samples processed")
        return jsonify(response), 200
        
    except Exception as e:
        duration = time.time() - start_time
        monitor.record_request('POST', '/predict/batch', 500, duration)
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    start_time = time.time()
    
    api_info = {
        'message': 'Palmer Penguins ML API',
        'version': '1.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'batch_predict': '/predict/batch (POST)',
            'model_info': '/model/info',
            'metrics': '/metrics'
        },
        'model_status': 'loaded' if model is not None else 'not loaded'
    }
    
    duration = time.time() - start_time
    monitor.record_request('GET', '/', 200, duration)
    
    return jsonify(api_info), 200

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    start_time = time.time()
    
    try:
        if model is None:
            raise Exception("Model not loaded")
        
        # Get basic model info
        info = {
            'model_type': type(model).__name__,
            'features': ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                        'body_mass_g', 'island_encoded', 'sex_encoded'],
            'target_classes': list(species_mapping.values()),
            'model_loaded': True
        }
        
        duration = time.time() - start_time
        monitor.record_request('GET', '/model/info', 200, duration)
        
        return jsonify(info), 200
        
    except Exception as e:
        duration = time.time() - start_time
        monitor.record_request('GET', '/model/info', 500, duration)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting ML serving API...")
        logger.info("Health check: http://localhost:5000/health")
        logger.info("Prediction: POST http://localhost:5000/predict")
        logger.info("Batch prediction: POST http://localhost:5000/predict/batch")
        logger.info("Metrics: http://localhost:5000/metrics")
        logger.info("Model info: http://localhost:5000/model/info")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Cannot start API without model")