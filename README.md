# ML Model Drift Detector & Auto-Retrainer

An end-to-end **MLOps pipeline** that detects data drift in deployed machine learning models and automatically triggers retraining to maintain performance over time.

---

## 1. Project Overview
Machine learning models degrade in real-world deployments due to **data drift** and **distribution shifts**.  
This project implements a **model monitoring system** that continuously compares incoming data against training data, detects statistically significant drift, and initiates **automated retraining** when necessary.

The system is designed to simulate a real production setup rather than a one-time offline experiment.

---

## 2. Drift Detection Mechanism
- Incoming data is treated as **new production data**
- Feature distributions are compared with training data using statistical divergence measures  
- A configurable **drift threshold** determines when the model is no longer reliable
- Drift detection is performed at the **feature level**, enabling fine-grained monitoring

This helps identify *when* and *why* a deployed model starts failing.

---

## 3. Automated Retraining Pipeline
- Once drift crosses the threshold, the system automatically:
  - Merges new data with historical data
  - Retrains the selected ML model
  - Saves the updated model version
- Ensures minimal manual intervention and supports **continuous learning**
- Modular design allows swapping models or drift metrics easily

This mimics **real-world MLOps retraining workflows** used in production systems.

---

## 4. Results & Visualizations
- Logs drift scores and retraining events for transparency
- Supports visualization of feature drift and retraining triggers  
- Enables comparison of model performance **before and after retraining**

> 📸 *Screenshots and visual examples will be added here to demonstrate drift detection and retraining in action.*

---

## Tech Stack
- Python
- Scikit-learn
- Pandas / NumPy
- Streamlit (for deployment & visualization)
- GitHub for version control

---

## Use Cases
- Monitoring deployed ML models
- Understanding real-world data drift
- Demonstrating MLOps and production ML workflows
- Resume-ready project for ML Engineer / MLOps roles
