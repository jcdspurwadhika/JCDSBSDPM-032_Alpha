# JCDSBSDPM-032_Alpha

# Customer Segmentation Framework using RFMS Analysis and Machine Learning

## Project Overview

This project develops an integrated customer segmentation framework using the Brazilian E-Commerce Public Dataset by Olist.

The project combines traditional RFMS (Recency, Frequency, Monetary, and Satisfaction) analysis with machine learning-based customer segmentation to provide more comprehensive customer insights for business decision-making.

Rather than replacing traditional segmentation, machine learning is used to complement RFMS by identifying behavioral patterns that may not be captured by rule-based approaches.

---

## Business Problem

Customer retention is one of the major challenges in e-commerce. Although companies often segment customers using traditional RFM analysis, customers within the same lifecycle stage may still exhibit different purchasing behaviors and customer experiences.

The objective of this project is to build a customer segmentation framework that helps businesses:

- Understand customer behavior
- Identify high-value customer groups
- Support targeted marketing strategies
- Improve customer retention
- Enhance customer experience

---

## Dataset

Dataset:
Brazilian E-Commerce Public Dataset by Olist

The dataset contains customer, order, payment, review, seller, and delivery information from an e-commerce platform.

---

## Methodology

The project consists of several stages:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Exploratory Data Analysis (EDA)
5. RFMS Segmentation
6. Machine Learning-Based Customer Segmentation
7. Comparison of Segmentation Approaches
8. Business Recommendations
9. Streamlit Deployment

---

## Final Model

The deployed prediction pipeline consists of:

- SimpleImputer (Median)
- RobustScaler
- Random Forest Classifier

Input Features:

- payment_value
- mean_review_score
- mean_delivery_days

Target:

- cluster_3f

The trained pipeline is saved as:

```
models/kmeans_pipeline.pkl
```

---

## Streamlit Application

The Streamlit application allows users to upload a customer-level dataset in CSV format.

The application will:

- Validate the uploaded dataset
- Extract the required features
- Predict customer segments
- Display prediction results
- Allow users to download the enriched dataset

### Intended Users

- Primary: CRM Teams
- Secondary: Marketing Analysts, Business Analysts, and Customer Experience Managers

---

## Repository Structure

```
customer-segmentation-framework/
│
├── notebooks/
│   └── Final_Project.ipynb
│
├── streamlit/
│   └── app.py
│
├── models/
│   └── kmeans_pipeline.pkl
│
├── sample_data/
│   └── sample_customer_level.csv
│
├── assets/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

---

## Results

The project demonstrates that:

- RFMS provides interpretable customer lifecycle segmentation.
- Machine learning complements RFMS by identifying behavioral patterns.
- Combining both approaches provides more actionable customer insights than using either method independently.

---

## Author

Purwadhika Data Science & Machine Learning Final Project

Author:
Canly Hansen Sudirman
Fauzan Hanindyawan Yusnidzar
