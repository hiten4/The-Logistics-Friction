# The-Logistics-Friction 
## Supply Chain Delay Oracle

Predicting delivery delays in e-commerce logistics to improve customer trust and operational efficiency.

---

## 🚀 Problem Statement

In large-scale e-commerce systems, inaccurate delivery estimates lead to:

- Customer dissatisfaction  
- Increased support costs  
- Reduced customer lifetime value  

This project aims to build a predictive system that **identifies orders likely to be delayed before shipment**, enabling proactive interventions.

---

## 🎯 Objective

- Predict whether an order will be delayed  
- Provide actionable insights for logistics and operations teams  
- Improve delivery estimation accuracy  

---

## 🧠 Business Impact

- Proactive customer communication  
- Better logistics prioritization  
- Reduced negative reviews and churn  
- Improved trust in delivery timelines  

---

## 📊 Dataset Overview

The dataset consists of multiple relational tables:

- `orders.csv`
- `order_items.csv`
- `customers.csv`
- `sellers.csv`
- (Additional supporting tables)

### Key Challenges

- Multi-table joins (complex relationships)
- Temporal dependencies (order → shipment → delivery)
- Geographic variability
- Missing and inconsistent data

---
---

## 🤖 Modelling & Evaluation

### Models Built
Four classification models were trained to predict `is_delayed`:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.7098 | 0.1689 |
| Random Forest | 0.7404 | 0.2021 |
| XGBoost ✅ | 0.7585 | 0.2397 |
| LightGBM | 0.7575 | 0.2336 |

> **Primary metric: PR-AUC** — used because the dataset is heavily imbalanced (only 7.7% delayed orders). Accuracy alone is misleading in such cases.

---

### 🏆 Best Model — XGBoost
XGBoost outperformed all other models with a **PR-AUC of 0.2397** — a 57% improvement over the Checkpoint 1 baseline of 0.153.

**Why XGBoost?**
- Builds trees sequentially — each tree learns from the mistakes of the previous one
- Handles class imbalance directly using `scale_pos_weight`
- Captures non-linear relationships between features (e.g. high freight + remote state = higher delay risk)

---

### 💼 Business Impact of the Model
- Model catches **~63% of real delays** before the package leaves the warehouse
- Enables **proactive customer notifications** — reducing complaints and negative reviews
- Gives the logistics team a **prioritised action list** of at-risk orders every day
- Every missed delay = lost customer trust. Every caught delay = a chance to recover it.

---

### ⚙️ How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Update the data path in `settings.py`**
```python
DATA_PATH = r"path\to\your\final_dataset.csv"
```

**3. Run individual models**
```bash
python".py files/logistic_regression_baseline.py"
python ".py files/model_2_random_forest.py"
python ".py files/model_3_xgboost.py"
python ".py files/model_4_lightgbm.py"
```

**4. Run full comparison + threshold tuning**
```bash
python ".py files/model_comparison.py"
```

---

