
# 📉 Netflix Customer Churn Prediction & Dynamic Pricing Optimization

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-f59e0b?style=for-the-badge)

An end-to-end machine learning pipeline for predicting customer churn and optimizing pricing strategies using Netflix subscription behavioral data. The project applies EDA, feature engineering, ensemble ML models, and an interactive Streamlit dashboard to support data-driven retention decisions.

---

## 📌 Project Overview

Customer churn is one of the most critical challenges for subscription-based businesses like Netflix. Even a small increase in churn rate can significantly reduce revenue. This project builds a **full ML pipeline** that:

- Analyzes behavioral and demographic patterns of 5,000 Netflix customers
- Identifies the strongest churn predictors through EDA and feature importance
- Trains and evaluates multiple classification models achieving **99.75% accuracy (XGBoost)**
- Builds an **interactive Streamlit dashboard** for non-technical stakeholders
- Applies pricing optimization analysis to support retention strategy recommendations

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| **Source** | Netflix Customer Churn Dataset |
| **Records** | 5,000 customers |
| **Features** | 14 variables |
| **Target** | `churned` (1 = churned, 0 = retained) |
| **Class Balance** | ~502 churned · ~498 retained (balanced) |

**Key Features:**
- **Demographics:** Age, Gender, Region
- **Subscription:** Subscription Type, Monthly Fee, Number of Profiles
- **Behavioral:** Watch Hours, Avg Watch Time Per Day, Last Login Days
- **Preferences:** Favorite Genre, Device, Payment Method

---

## 🧠 Data & ML Concepts Demonstrated

| Concept | Implementation |
|---|---|
| **EDA** | Missing value analysis, distribution plots, outlier detection, correlation heatmap |
| **Feature Engineering** | One-hot encoding, StandardScaler, outlier treatment, feature importance |
| **ML Classification** | Logistic Regression, Random Forest, XGBoost, Decision Tree, KNN, Naive Bayes |
| **Model Evaluation** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Deep Learning** | Neural Network classifier for churn prediction |
| **Pricing Optimization** | Dynamic pricing analysis based on churn probability scores |
| **Data Visualization** | Matplotlib, Seaborn — 10+ chart types |
| **Dashboard** | Interactive Streamlit app for stakeholder exploration |

---

## 🔍 Key Findings from EDA

**Strongest Churn Predictors:**
- 📅 **Last Login Days** — longest inactivity = highest churn risk
- 📺 **Watch Hours** — lower engagement = higher churn probability  
- ⏱️ **Avg Watch Time Per Day** — top feature importance (0.40 score)

**Other Insights:**
- Balanced class distribution (50/50) — no SMOTE needed
- Age and Monthly Fee show weak churn correlation
- Behavioral engagement variables far outperform demographic features

---

## 🤖 Model Performance

| Model | Validation Accuracy |
|---|---|
| **XGBoost** | **99.75%** 🏆 |
| Random Forest | 99.58% |
| Decision Tree | 98.76% |
| KNN | 94.77% |
| Logistic Regression | 94.68% |
| Naive Bayes | 72.21% |

**Baseline Logistic Regression Results:**
- Accuracy: **89.5%**
- Precision: 0.90 · Recall: 0.90 · F1: 0.89
- True Positives: 459 · True Negatives: 436
- False Negatives (missed churners): only 43

---

## 🏗️ Pipeline Architecture

```
Raw Netflix Dataset (5,000 customers)
        ↓
Data Cleaning & Validation
(duplicates, type fixes, range checks, outlier treatment)
        ↓
Exploratory Data Analysis
(distributions, correlations, churn split analysis)
        ↓
Feature Engineering
(one-hot encoding, StandardScaler, feature selection)
        ↓
Model Training & Evaluation
(Logistic Regression → Random Forest → XGBoost → Neural Net)
        ↓
Feature Importance Analysis
(Random Forest importance scores)
        ↓
Churn Probability Scoring
(risk-level classification for each customer)
        ↓
Streamlit Dashboard
(interactive visualization for stakeholders)
```

---

## 📁 Project Structure

```
customer-churn/
├── data/
│   └── netflix_churn.csv          # Raw dataset
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb     # Data cleaning & feature engineering
│   └── 03_Modeling.ipynb          # Model training & evaluation
├── src/
│   ├── preprocessing.py           # Data pipeline functions
│   ├── models.py                  # ML model training
│   └── evaluation.py             # Metrics & visualization
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas · NumPy |
| **Visualization** | Matplotlib · Seaborn |
| **ML Models** | Scikit-learn · XGBoost |
| **Deep Learning** | PyTorch |
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
- [ ] PCA dimensionality reduction after one-hot encoding
- [ ] Churn probability scoring system for risk-level prioritization
- [ ] Customer segmentation analysis (high-risk behavioral groups)
- [ ] Dynamic pricing optimization model
- [ ] Deploy Streamlit dashboard publicly

---

## 👥 Author

**Hiral Rana** 

---

## 📄 License
This project is for educational and portfolio purposes.
