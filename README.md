# Project Title
Why Did My Model Fail? – Automated ML Diagnosis System

Author: Suhani Bode

Affiliation: Rashtrasant Tukadoji Maharaj Nagpur University

Date: June 2026

# Abstract

Machine learning models often fail due to issues such as poor data quality, class imbalance, overfitting, and weak feature selection. This project, "Why Did My Model Fail?", is an automated ML diagnostic platform that analyzes datasets, trains multiple machine learning models, compares their performance, and identifies possible reasons behind model failure. The system performs preprocessing, model evaluation, feature importance analysis, and visualization through an interactive dashboard. Testing on the Adult Income dataset (32,560 records) showed that Gradient Boosting achieved the best performance with 86.5% accuracy and a ROC-AUC score of 0.9224. The platform helps students and developers understand model behavior and improve machine learning performance.

# Introduction

Machine learning models are widely used in real-world applications, but understanding why a model performs poorly can be difficult. Problems such as overfitting, underfitting, class imbalance, and irrelevant features can reduce model accuracy. This project aims to provide an automated solution that analyzes datasets, compares models, and explains performance through visual reports and diagnostic insights.

# Literature Review

Several AutoML and machine learning platforms such as Google AutoML, H2O.ai, and Azure ML provide automated model training and evaluation. Explainable AI (XAI) techniques focus on understanding model predictions through feature importance and performance metrics. This project combines these ideas into a simple and user-friendly diagnostic platform.

# Methodology

The user uploads a dataset, which is preprocessed and analyzed. Multiple machine learning models are trained and evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Cross-Validation Score. The system automatically selects the best-performing model and performs feature importance analysis. Results are displayed through an interactive dashboard and exported as reports.

# Implementation

Programming Language:
- Python

Frameworks/Libraries:
- Flask
- Pandas
- NumPy
- Scikit-learn
- Plotly
- ReportLab

Tools Used:
- VS Code
- GitHub
- Vercel

# Results and Discussion

Dataset:
- Adult Income Dataset
- 32,560 Rows
- 15 Features

Best Model:
- Gradient Boosting

Performance Metrics:
- Test Accuracy: 86.5%
- Train Accuracy: 86.5%
- Cross-Validation Score: 85.5%
- ROC-AUC Score: 0.9224

Model Comparison:
- Gradient Boosting: 86.5%
- Random Forest: 85.8%
- Decision Tree: 85.7%
- KNN: 82.9%
- Logistic Regression: 82.7%
- SVM: 81.5%
- Naive Bayes: 80.7%

The results show that Gradient Boosting performed best with high accuracy and strong classification capability. Feature importance analysis highlighted the most influential factors affecting predictions.

# Limitation

- Supports only structured tabular datasets.
- Limited hyperparameter tuning.
- No support for deep learning models.
- Processing time increases for very large datasets.

# Future Scope

- Integrate SHAP and LIME for Explainable AI.
- Add deep learning model support.
- Implement automated hyperparameter tuning.
- Support image, text, and time-series datasets.
- Enable real-time model monitoring.

# Conclusion

This project provides an automated machine learning diagnostic system that helps users understand model performance and identify possible reasons for failure. By combining model comparison, feature analysis, and visual reporting, the platform simplifies the machine learning workflow and improves decision-making for students and developers.

# References

[1] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, 2016.

[2] F. Pedregosa et al., "Scikit-Learn: Machine Learning in Python," JMLR, 2011.

[3] J. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," Annals of Statistics, 2001.

[4] https://scikit-learn.org

[5] https://flask.palletsprojects.com

[6] https://pandas.pydata.org
