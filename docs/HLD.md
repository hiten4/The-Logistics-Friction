# High-Level Design (HLD)

## 1. Overview

This system predicts whether an e-commerce order will be delayed using historical logistics data.  
The goal is to enable proactive intervention and improve customer experience.

---

## 2. Business Objective

- Predict delivery delays before shipment  
- Reduce customer dissatisfaction  
- Enable operational prioritization  

---

## 3. System Architecture

The system follows a structured ML pipeline:

Raw CSV Files  
↓  
Data Ingestion  
↓  
Data Preprocessing  
↓  
Feature Engineering  
↓  
Feature Store / Dataset  
↓  
Model Training  
↓  
Model Evaluation  
↓  
Model Deployment  
↓  
Prediction (Batch / Real-time)  

---

## 4. Component Description

### 4.1 Data Sources
- Multiple CSV files representing different entities:
  - Orders
  - Order Items
  - Customers
  - Sellers

---

### 4.2 Data Ingestion
- Load raw CSV data into the system
- Ensure schema consistency

---

### 4.3 Data Preprocessing
- Handle missing values
- Convert data types (especially timestamps)
- Remove duplicates and inconsistencies

---

### 4.4 Feature Engineering
- Generate meaningful features:
  - Time-based features (delivery duration, delays)
  - Order-level features (item count, cost)
  - Derived target variable (delay flag)

---

### 4.5 Feature Store / Dataset
- Consolidated dataset ready for modeling
- Stored as a processed file for reuse

---

### 4.6 Model Training
- Train ML models on processed dataset
- Experiment with multiple algorithms

---

### 4.7 Model Evaluation
- Evaluate using:
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

---

### 4.8 Model Deployment
- Deploy trained model for predictions
- Supports:
  - Batch inference
  - Real-time API

---

### 4.9 Prediction Layer
- Output:
  - `is_delayed = 0 / 1`
- Used by business systems for decision-making

---

## 5. Assumptions

- Historical data patterns reflect future behavior  
- Data quality is sufficient after preprocessing  

---

## 6. Risks

- Data inconsistency across sources  
- Class imbalance  
- Temporal leakage  
- Changing logistics conditions  

---

## 7. Summary

This HLD defines a scalable and modular ML pipeline that transforms raw logistics data into actionable delay predictions.
