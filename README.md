# Medical Equipment Risk Predictor API

This repository contains a Machine Learning API built with **Flask** and designed to be production-ready using **Docker** and **Gunicorn**. It serves machine learning models to predict medical equipment failure and classify risk based on either time-series sensor data or tabular device features.

This is meant to be run as a microservice and can easily be connected to a primary backend (e.g., a Java application) via HTTP POST requests.

---

## 📂 Directory Structure

The project has been organized into logical directories so it's easy to understand:

- **`ml_api/`**: The core Flask application code.
  - `routes.py`: Contains the API endpoint definitions (`/predict/lstm` and `/predict/rf`).
  - `lstm_service.py` & `rf_service.py`: Services that load the models and run predictions.
  - `config.py`: Environment variable configurations.
- **`artifacts/`**: Contains the pre-trained `.keras`, `.joblib`, and `.npy` model files and scalers required for inference.
- **`data/`**: Datasets used for training or testing.
  - `raw/`: Raw CSV files (e.g., `SmouhaMedicalCenter.csv`).
  - `processed/`: Cleaned and processed data files.
- **`notebooks/`**: Jupyter Notebooks (`.ipynb`) used for data exploration, model training, and preprocessing.
- **`scripts/`**: Python scripts for generating synthetic data, training models, and testing endpoints locally.
- **`docs/`**: Documentation, explanations, presentations, and diagrams.
- **`archive/`**: Old, experimental, or unorganized files moved out of the main project flow.

---

## 🚀 How to Run (Production Ready)

The application is containerized and optimized for production using `docker-compose`. 

1. **Build and start the container:**
   ```bash
   docker-compose up --build -d
   ```
2. **Verify it's running:**
   Navigate to `http://localhost:5000/health` in your browser or make a GET request. You should receive a `{"status": "ok"}` response.

> **Note:** The Docker container runs securely as a non-root user and uses Gunicorn as a production WSGI server to handle multiple requests concurrently.

---

## 📡 API Reference (For Backend Integration)

Your main backend (e.g., Java/Spring Boot) can connect to this service by sending standard HTTP `POST` requests with JSON payloads.

### 1. LSTM Sequence Prediction
**Endpoint**: `POST /predict/lstm`
**Description**: Predicts the probability of failure based on a sequence of 30 time-steps.

**Request Body (JSON):**
```json
{
  "sequence": [
    [1.2, 0.5, 3.4], 
    [1.1, 0.6, 3.3],
    ... // Must contain exactly 30 arrays (rows)
  ]
}
```

**Response (JSON):**
```json
{
  "probability": 0.85,
  "prediction": 1
}
```

### 2. Random Forest Risk Classification
**Endpoint**: `POST /predict/rf`
**Description**: Predicts the risk class (e.g., Low, Medium, High) based on static/tabular device features.

**Request Body (JSON):**
```json
{
  "features": {
      "Device_Type": "MRI",
      "Age": 5,
      "MTBF": 1200
  }
}
```

**Response (JSON):**
```json
{
  "confidence_score": 0.92,
  "predicted_class": 2,
  "predicted_label": "High Risk"
}
```
