# Palmer Penguins ML Pipeline with Monitoring

End-to-end machine learning pipeline for penguin species classification with MLflow tracking, REST API serving, and Prometheus/Grafana monitoring.


## Features

- **Multiple ML Models**: Random Forest, Logistic Regression, SVM
- **MLflow Integration**: Experiment tracking and model registry
- **REST API**: Flask-based prediction service
- **Monitoring Stack**: Prometheus metrics + Grafana dashboards
- **System Monitoring**: CPU, memory, disk usage tracking
- **Prediction Metrics**: Request counts, latencies, accuracy

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python modelling.py
```

This will:
- Load preprocessed penguin data
- Train Random Forest, Logistic Regression, and SVM models
- Track experiments in MLflow
- Register best model

### 3. Start Monitoring Stack

```bash
docker-compose up -d
```

Services:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Node Exporter**: http://localhost:9100

### 4. Start ML API

```bash
python ml_serving.py
```

API endpoints:
- **Health**: GET http://localhost:5000/health
- **Predict**: POST http://localhost:5000/predict
- **Batch**: POST http://localhost:5000/predict/batch
- **Metrics**: GET http://localhost:5000/metrics

### 5. Start Custom Monitoring

```bash
python prometheus_exporter.py
```

Custom metrics server: http://localhost:8000/metrics

## API Usage

### Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "island_encoded": 0,
    "sex_encoded": 1
  }'
```

Response:
```json
{
  "predicted_species": "Adelie",
  "confidence": 0.95,
  "probabilities": {
    "Adelie": 0.95,
    "Chinstrap": 0.03,
    "Gentoo": 0.02
  },
  "prediction_time": 0.012
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "island_encoded": 0,
        "sex_encoded": 1
      }
    ]
  }'
```

## Feature Encoding

| Feature | Type | Encoding |
|---------|------|----------|
| bill_length_mm | Numeric | Raw value |
| bill_depth_mm | Numeric | Raw value |
| flipper_length_mm | Numeric | Raw value |
| body_mass_g | Numeric | Raw value |
| island_encoded | Categorical | 0=Torgersen, 1=Biscoe, 2=Dream |
| sex_encoded | Categorical | 0=Female, 1=Male |

## Monitoring & Metrics

### Prometheus Metrics

**System Metrics:**
- `system_cpu_usage_percent`: CPU usage
- `system_memory_usage_percent`: Memory usage  
- `system_disk_usage_percent`: Disk usage

**ML Metrics:**
- `ml_requests_total`: API request counts by method/endpoint/status
- `ml_request_duration_seconds`: Request latency histogram
- `ml_predictions_total`: Prediction counts by model/species
- `ml_model_accuracy`: Current model accuracy

### Grafana Setup

1. Open http://localhost:3000 (admin/admin)
2. Add Prometheus data source: `http://prometheus:9090`
3. Import dashboard or create panels with queries:
   - System CPU: `system_cpu_usage_percent`
   - API Requests: `rate(ml_requests_total[5m])`
   - Prediction Distribution: `ml_predictions_total`

## MLflow Tracking

Access MLflow UI:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

View at http://localhost:5000 for:
- Experiment comparisons
- Model metrics and parameters
- Model artifacts and versions

## Testing

### Run Inference Tests

```bash
python inference.py
```

This simulates predictions and generates monitoring data.

### Manual API Testing

```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model/info

# Metrics
curl http://localhost:5000/metrics
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   Flask API     │───▶│   MLflow Model  │
│                 │    │  (ml_serving.py)│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Prometheus    │
                       │   Exporter      │
                       └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Grafana      │◀───│   Prometheus    │◀───│  Node Exporter  │
│   Dashboard     │    │    Server       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Configuration

### Environment Variables

```bash
export MLFLOW_TRACKING_URI="file:./mlruns"
export FLASK_ENV="production"
export PROMETHEUS_PORT="8000"
```

### Docker Network Access

For services running on host machine, Prometheus uses `host.docker.internal` to scrape metrics from:
- ML API (port 5000)
- Custom metrics (port 8000)

## Troubleshooting

**Connection refused errors:**
- Ensure all services are running
- Check Docker network configuration
- Verify ports are not blocked

**Model not loading:**
- Check MLflow model registry
- Verify model artifacts exist
- Check file permissions

**No metrics in Grafana:**
- Verify Prometheus data source URL: `http://prometheus:9090`
- Check if metrics endpoint returns data
- Ensure correct query syntax

## Dependencies

Core packages:
- pandas, numpy, scikit-learn: ML pipeline
- mlflow: Experiment tracking
- flask: API framework  
- prometheus_client: Metrics export
- psutil: System monitoring

See `requirements.txt` for complete list with versions.

## License

This project is for educational purposes demonstrating MLOps practices with monitoring and observability.