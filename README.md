# ML Model Drift Detector & Auto-Retrainer

A production-style **MLOps pipeline** that detects **data drift** in deployed machine learning models and automatically retrains them to maintain reliable performance over time.

---

## 1. What is Model Drift? (With Example)

In real-world deployments, machine learning models often fail **not because the model is bad**, but because the **data changes over time**.  
This phenomenon is known as **model drift**.

### Example:
Suppose a model is trained to predict **obesity risk** using features like:
- Age
- BMI
- Daily calorie intake
- Physical activity level

Initially, the model is trained on historical data.  
Over time:
- User lifestyle patterns change  
- Diet trends evolve  
- Input distributions shift  

Even though the model logic remains the same, the **incoming data distribution no longer matches the training data**, causing prediction accuracy to drop.

This mismatch between **old data (training)** and **new data (production)** is called **data drift**.

---

## 2. How This Project Solves Drift

This project simulates a **real-world deployed ML system** and introduces an automated solution to handle drift:

1. Incoming data is treated as **live production data**
2. Feature distributions of new data are continuously compared with training data
3. If the statistical difference exceeds a defined threshold:
   - Drift is detected
   - The model is marked as unreliable
4. The system automatically triggers **model retraining** using updated data

This removes the need for manual monitoring and retraining.

---

## 3. What Is Implemented in This Project

### Drift Detection
- Feature-level comparison between training and incoming data
- Statistical drift scoring with configurable thresholds
- Clear identification of **when drift occurs**

### Automated Retraining
- Automatic merging of historical and new data
- Model retraining pipeline triggered only when drift is detected
- Updated model is saved and reused for future predictions

### MLOps-Oriented Design
- Modular pipeline (drift detection, retraining, logging separated)
- Easily extendable to different datasets or ML models
- Simulates real production ML lifecycle instead of offline experiments

---

## 4. Application & Demo

The project includes an interactive **Streamlit application** that allows:
- Uploading new input data
- Visualizing drift detection results
- Observing retraining triggers and updated predictions

🔗 **Live App Link:**  
👉 _Add your deployed Streamlit / Hugging Face / Localhost link here_

> Example: https://huggingface.co/spaces/your-username/model-drift-detector

📸 Screenshots of the app interface and drift visualization will be added here.

---

## Tech Stack
- Python
- Scikit-learn
- Pandas / NumPy
- Streamlit
- Git & GitHub

---

## Why This Project Matters
- Demonstrates **real-world ML deployment challenges**
- Shows understanding of **MLOps, monitoring, and automation**
- Goes beyond basic model training into **production ML systems**
- Resume-ready project for **ML Engineer / MLOps / Data Science** roles
