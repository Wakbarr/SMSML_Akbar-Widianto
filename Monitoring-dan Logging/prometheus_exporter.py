import time
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import requests
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total ML API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('ml_request_duration_seconds', 'ML API request duration')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage')
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total predictions made', ['model', 'species'])

class MLMonitor:
    def __init__(self, model_endpoint='http://localhost:5000'):
        self.model_endpoint = model_endpoint
        self.running = False
        
    def start_monitoring(self):
        """Start the monitoring process"""
        self.running = True
        
        # Start Prometheus metrics server
        start_http_server(8000)
        logger.info("Prometheus metrics server started on port 8000")
        
        # Start system monitoring thread
        monitoring_thread = threading.Thread(target=self._monitor_system_metrics)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        logger.info("ML monitoring started")
        
    def _monitor_system_metrics(self):
        """Monitor system metrics continuously"""
        while self.running:
            try:
                # Monitor CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                SYSTEM_CPU_USAGE.set(cpu_percent)
                
                # Monitor memory usage
                memory = psutil.virtual_memory()
                SYSTEM_MEMORY_USAGE.set(memory.percent)
                
                # Monitor disk usage (root partition)
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                SYSTEM_DISK_USAGE.set(disk_percent)
                
                logger.info(f"System metrics - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk_percent:.1f}%")
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                time.sleep(10)
    
    def record_request(self, method, endpoint, status_code, duration):
        """Record API request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
        REQUEST_DURATION.observe(duration)
        
    def record_prediction(self, model_name, predicted_species):
        """Record prediction metrics"""
        PREDICTION_COUNT.labels(model=model_name, species=predicted_species).inc()
        
    def update_model_accuracy(self, accuracy):
        """Update model accuracy metric"""
        MODEL_ACCURACY.set(accuracy)
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        logger.info("ML monitoring stopped")

# Example usage and testing
if __name__ == "__main__":
    monitor = MLMonitor()
    monitor.start_monitoring()
    
    # Simulate some metrics for testing
    import random
    
    try:
        while True:
            # Simulate API requests
            methods = ['GET', 'POST']
            endpoints = ['/predict', '/health', '/metrics']
            status_codes = [200, 201, 400, 500]
            
            method = random.choice(methods)
            endpoint = random.choice(endpoints)
            status = random.choice(status_codes)
            duration = random.uniform(0.1, 2.0)
            
            monitor.record_request(method, endpoint, status, duration)
            
            # Simulate predictions
            models = ['Random_Forest', 'Logistic_Regression', 'SVM']
            species = ['Adelie', 'Chinstrap', 'Gentoo']
            
            model = random.choice(models)
            predicted_species = random.choice(species)
            
            monitor.record_prediction(model, predicted_species)
            
            # Simulate model accuracy updates
            accuracy = random.uniform(0.85, 0.95)
            monitor.update_model_accuracy(accuracy)
            
            logger.info(f"Recorded: {method} {endpoint} {status} ({duration:.2f}s)")
            logger.info(f"Prediction: {model} -> {predicted_species}")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        logger.info("Stopping monitoring...")
        monitor.stop_monitoring()