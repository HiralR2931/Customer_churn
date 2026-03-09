# 📉 Netflix Customer Churn Prediction & Dynamic Pricing Optimization

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-f59e0b?style=for-the-badge)

An end-to-end machine learning pipeline for predicting customer churn and optimizing pricing strategies using Netflix subscription behavioral data. The project spans the full data lifecycle — from exploratory analysis and feature engineering through ensemble model comparison and interactive dashboard delivery.

---

## 📌 Project Overview

Customer churn is one of the most critical challenges for subscription-based businesses. Even a small increase in churn rate can significantly reduce recurring revenue, making it essential to understand why customers leave and how to intervene early.

This project builds a **complete ML pipeline** that:

- Processes and validates **5,000 Netflix customer records** across 14 features
- Performs comprehensive EDA with **10+ visualization types** to uncover churn drivers
- Engineers features through one-hot encoding, standard scaling, and outlier treatment
- Trains and benchmarks **6 classification models**, achieving **99.75% accuracy (XGBoost)**
- Extracts feature importance scores to identify the strongest churn predictors
- Delivers an **interactive Streamlit dashboard** for non-technical stakeholder exploration

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| **Source** | [Netflix Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset) |
| **Records** | 5,000 customers |
| **Features** | 14 variables (mixed numeric & categorical) |
| **Target** | `churned` (1 = churned, 0 = retained) |
| **Class Balance** | ~502 churned · ~498 retained (balanced — no SMOTE required) |

**Feature Categories:**

| Category | Variables |
|---|---|
| **Demographics** | Age, Gender, Region |
| **Subscription** | Subscription Type, Monthly Fee, Number of Profiles |
| **Behavioral** | Watch Hours, Avg Watch Time Per Day, Last Login Days |
| **Preferences** | Favorite Genre, Device, Payment Method |

---

## 🔬 Data Processing Pipeline

### Data Cleaning & Validation
- Removed duplicate records to prevent training bias
- Validated data types and corrected numeric/categorical formatting
- Applied range validation on age, login recency, and subscription fields
- Corrected decimal-point entry errors in `avg_watch_time_per_day` (values > 10 hrs ÷ 10)
- Standardized category labels (whitespace removal, consistent formatting)
- Confirmed zero missing values — no imputation required

### Feature Engineering
- **One-Hot Encoding** via `pd.get_dummies(drop_first=True)` for all categorical variables (gender, region, device, subscription type, payment method, genre)
- **StandardScaler** normalization on numerical features for gradient-based models
- **Outlier Treatment** — flagged extreme watch behavior values (80–98 hrs) for potential capping/log-transformation in advanced stages; retained for baseline modeling
- Consistent encoding applied across train/test splits to prevent category mismatch

### Data Leakage Prevention
- Scaling parameters fitted on training data only, then applied to test data
- Stratified 80/20 train-test split preserving churn class balance

---

## 🔍 Key EDA Findings

### Strongest Churn Predictors
| Feature | Signal | Correlation with Churn |
|---|---|---|
| **Avg Watch Time Per Day** | Top feature importance (0.40 score) | −0.27 |
| **Watch Hours** | Lower engagement → higher churn | −0.48 |
| **Last Login Days** | Longer inactivity → highest churn risk | +0.47 |

### Other Insights
- **Balanced classes** (50/50 split) — accuracy is a meaningful metric without resampling
- **Age and Monthly Fee** show weak churn correlation (−0.01 to −0.00)
- **Behavioral engagement variables** far outperform demographics as predictors
- **Low multicollinearity** across features — each contributes distinct predictive signal
- **Categorical features** (subscription type, region, device, genre, payment method) are all evenly distributed — no representation bias

---

## 🤖 Model Comparison

| Model | Validation Accuracy |
|---|---|
| **XGBoost** | **99.75%** 🏆 |
| Random Forest | 99.58% |
| Decision Tree | 98.76% |
| KNN | 94.77% |
| Logistic Regression | 94.68% |
| Naive Bayes | 72.21% |

### Baseline Model — Logistic Regression

| Metric | Class 0 (Retained) | Class 1 (Churned) |
|---|---|---|
| **Precision** | 0.91 | 0.88 |
| **Recall** | 0.88 | 0.91 |
| **F1-Score** | 0.89 | 0.90 |

- **Overall Accuracy:** 89.5%
- **True Positives:** 459 · **True Negatives:** 436
- **False Negatives (missed churners):** only 43

### Why Tree-Based Models Dominate
Ensemble models (XGBoost, Random Forest) capture nonlinear relationships and complex feature interactions that linear models cannot. Naive Bayes underperformed due to violations of its feature independence assumption among correlated engagement variables.

---

## 🏗️ Pipeline Architecture

```
Raw Netflix Dataset (5,000 records, 14 features)
        │
        ▼
Data Cleaning & Validation
(duplicates, type fixes, range checks, decimal correction)
        │
        ▼
Exploratory Data Analysis
(distributions, correlations, churn splits, outlier detection)
        │
        ▼
Feature Engineering
(one-hot encoding, StandardScaler, train/test split)
        │
        ▼
Model Training & Benchmarking
(Logistic Regression → KNN → Decision Tree → Random Forest → XGBoost → Neural Net)
        │
        ▼
Feature Importance & Evaluation
(Random Forest importance scores, confusion matrix, classification report)
        │
        ▼
Churn Probability Scoring
(risk-level classification per customer)
        │
        ▼
Streamlit Dashboard
(interactive visualization for stakeholders)
```

---

## 📁 Project Structure

```
customer-churn/
├── data/
│   └── netflix_churn.csv              # Raw dataset (5,000 records)
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb         # Data cleaning & feature engineering
│   └── 03_Modeling.ipynb              # Model training & evaluation
├── src/
│   ├── preprocessing.py               # Data pipeline functions
│   ├── models.py                      # ML model training
│   └── evaluation.py                  # Metrics & visualization
├── dashboard/
│   └── app.py                         # Streamlit dashboard
├── reports/
│   └── DataCollection_Report.pdf      # Milestone report with EDA & results
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas · NumPy |
| **Visualization** | Matplotlib · Seaborn (10+ chart types) |
| **ML Models** | Scikit-learn · XGBoost |
| **Deep Learning** | PyTorch (Neural Network classifier) |
| **Dashboard** | Streamlit |
| **Environment** | Jupyter Notebook · Google Colab |

---

## 🚀 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/HiralR2931/customer-churn.git
cd customer-churn

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks
jupyter notebook

# Launch Streamlit dashboard
streamlit run dashboard/app.py
```

---

## 📈 Next Steps

- [ ] Hyperparameter tuning (GridSearchCV) for XGBoost and Random Forest
- [ ] PCA dimensionality reduction to manage post-encoding feature expansion
- [ ] Churn probability scoring system for risk-level customer prioritization
- [ ] Customer segmentation analysis (identify high-risk behavioral groups)
- [ ] Dynamic pricing optimization model based on churn probability scores
- [ ] Deploy Streamlit dashboard publicly

---

## 👥 Author

**Hiral Rana**
MS Data Analytics Engineering · Northeastern University

---

## 📄 License

This project is for educational and portfolio purposes.
---