# UCI-Heart-Disease

This project aims to analyze the heart disease dataset from the UCI Machine Learning Repository. We focus on predicting the likelihood of heart disease and clustering the data to identify patterns.

## Project Goals

- **Predict Heart Disease Risk:** Develop a model to accurately predict the likelihood of heart disease in patients based on clinical features.
- **Identify Data Patterns:** Use clustering techniques to discover patterns and group similar patient data, providing insights into different risk categories.
- **Enhance Decision-Making:** Provide healthcare professionals with tools to identify high-risk patients and tailor interventions.
- **Improve Model Performance:** Continuously refine models to achieve higher accuracy and reliability in predictions.

## Project Objectives

- **Accurate Prediction:** Develop a reliable model to predict the risk of heart disease using clinical data.
- **Data Clustering:** Identify and analyze patterns through clustering techniques to categorize patients into distinct groups.
- **Insight Generation:** Provide actionable insights that can assist healthcare practitioners in understanding risk factors.
- **Model Optimization:** Continuously improve model performance through iterative testing and validation.

## Problem Statement

- Heart disease remains a leading cause of morbidity and mortality worldwide. Early detection and intervention are crucial in reducing the impact of heart disease. However, accurately identifying patients at risk is          challenging due to the complexity of risk factors involved.

## Solution Approach

1. **Data Exploration and Preprocessing:**
   - Perform Exploratory Data Analysis (EDA) to understand data distribution and relationships.
   - Clean and preprocess data by handling missing values and normalizing features.

2. **Predictive Modeling:**
   - Implement machine learning algorithms such as Logistic Regression and Support Vector Machines (SVM) to predict heart disease risk.
   - Evaluate models based on accuracy, precision, and recall.

3. **Clustering Analysis:**
   - Apply K-Means clustering to segment the data into distinct groups.
   - Analyze clusters to identify patterns and potential risk categories.

4. **Model Optimization:**
   - Fine-tune model parameters and perform cross-validation to improve performance.
   - Test various algorithms to ensure the robustness of predictions and clusters.

5. **Evaluation Metrics:**
  - Accuracy, Precision, Recall, F1-score 
  
## Data Sources & Overview

**Data Origin:** [Kaggle UCI Heart Disease](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data)

**The dataset includes clinical features from patients, such as:**
  - id (Unique id for each patient)
  - age (Age of the patient in years)
  - origin (place of study)
  - sex (Male/Female)
  - cp chest pain type ([typical angina, atypical angina, non-anginal, asymptomatic])
  - trestbps resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
  - chol (serum cholesterol in mg/dl)
  - fbs (if fasting blood sugar > 120 mg/dl)
  - restecg (resting electrocardiographic results) Values: [normal, stt abnormality, lv hypertrophy]
  - thalach: maximum heart rate achieved
  - exang: exercise-induced angina (True/ False)
  - oldpeak: ST depression induced by exercise relative to rest
  - slope: the slope of the peak exercise ST segment
  - ca: number of major vessels (0-3) colored by fluoroscopy
  - thal: [normal; fixed defect; reversible defect]
  - num: the predicted attribute

## Technologies Used

- **Programming & Libraries:** Python, Numpy, Pandas, Scikit-learn, Matplotlib, Seaborn, Plotly.
- **Modeling Approaches:** Logistic Regression, Decision Tree, Random Forest, SVM, K-Means Clustering, Hierarchical Clustering.
- **Version Control & Collaboration:** Git, GitHub Projects.
  
## How to Run

1. Clone the repository to your computer.
2. Set up a Python environment.
3. Download the dataset from Kaggle and place files in the project directory.
4. Install dependencies such as numpy, pandas, scikit-learn, matplotlib, seaborn, and others.
5. Run The analysis: Execute the Notebook to perform predictions and clustering

