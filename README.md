# Predicting Hypertension Risk: A Machine Learning Approach

# Author: Teoh Leng Wah

# Executive summary
- This capstone project aims to develop an accurate and efficient model for predicting hypertension risk based on various health metrics and risk factors. By comparing different machine learning techniques, we explore the most effective approach for this crucial healthcare challenge. Our study focuses on K-Means Clustering, Principal Component Analysis (PCA), linear regression, ridge regression, lasso regression, and Sequential Feature Selection (SFS) to identify the best method for predicting hypertension risk.

# Rationale
- Hypertension is a leading cause of cardiovascular diseases worldwide. Early identification of individuals at high risk of developing hypertension is crucial for timely intervention and prevention. This project leverages various machine learning techniques to create predictive models that can assist healthcare professionals in identifying at-risk individuals, potentially leading to more targeted preventive measures and improved patient outcomes.
  
# Research Question
- The study addresses the following key research questions:
    - Which machine learning technique (K-Means Clustering, PCA, linear regression, ridge regression, lasso regression, SFS) yields the highest accuracy in predicting hypertension risk?
    - What are the most important features or risk factors contributing to hypertension risk prediction?
    - How can we optimize the selected machine learning model to improve its performance and generalizability?

# Data Sources
- Utilized the "Hypertension-risk-model-main.csv" dataset, which contains various health-related features including:

    - Demographic information (age, gender)
    - Lifestyle factors (smoking habits)
    - Clinical measurements (blood pressure, cholesterol levels, BMI, heart rate, glucose levels)
    - Medical history (diabetes, use of BP medication)

- The dataset comprises 4,240 entries with 13 columns, containing both integer and float data types.

  <img width="511" alt="Screenshot 2024-07-01 at 8 37 53 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/9456d8c9-3156-4031-bd75-0fc7da4b4e7c">

  <img width="774" alt="Screenshot 2024-07-01 at 8 37 32 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/55878198-1bbe-4d12-a4ea-61084034ceab">

  <img width="706" alt="Screenshot 2024-07-01 at 8 41 33 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/03b1ba4e-2bfa-4614-8bfa-20e3305bc4d0">



# Methodology
- Our approach followed these key steps:

1. Data Preprocessing:

    - Handled missing values in several columns (cigsPerDay, BPMeds, totChol, BMI, heartRate, glucose)
    - Encoded categorical variables
    - Scaled numerical features


2. Exploratory Data Analysis (EDA):

    - Analyzed the distribution of the target variable (Risk)
    - Explored relationships between features and hypertension risk
    - Created visualizations to understand feature interactions and distributions


3. Feature Selection and Engineering:

    - Applied Principal Component Analysis (PCA) for dimensionality reduction
    - Utilized Sequential Feature Selection (SFS) to identify optimal feature subsets


4. Model Development:

    - Split data into training and testing sets
    - Implemented multiple models:

      a) K-Mans Clustering
      b) Linear Regression
      c) Ridge Regression
      d) Lasso Regression

5. Model Evaluation:

    - Utilized cross-validation to ensure robust performance estimates
    - Assessed models using appropriate metrics (e.g., MSE, RMSE for regression tasks)
    - Compared the performance of different techniques


6. Hyperparameter Tuning:

    - Performed GridSearchCV on the best-performing model to optimize its parameters


7. Interpretation and Insights:

    - Analyzed feature importance
    - Interpreted model results in the context of hypertension risk prediction

# Results

- Our analysis yielded the following key findings:

1. Model Performance:

<img width="255" alt="Screenshot 2024-07-01 at 9 20 31 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/08a3c522-b58c-4e50-a093-1473c5e9d6c6">

2. Key Predictive Features:

  a) Systolic Blood Pressure (sysBP)
  b) Age
  c) Body Mass Index (BMI)
  d) Diastolic Blood Pressure (diaBP)
  e) Total Cholesterol (totChol)

3. Risk Factors:

  a) Age is strongly correlated with increased hypertension risk
  b) Males have a slightly higher proportion of high-risk cases
  c) Higher BMI is associated with increased risk
  d) Current smokers and individuals with diabetes show higher risk


# Dataset Characteristics:

- Imbalanced dataset: 68.81% low risk, 31.19% high risk
- This imbalance was addressed during model development

  <img width="695" alt="Screenshot 2024-07-01 at 8 22 51 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/172c2581-4446-4ba6-97e0-a9ce06ca1726">

## EDA Insights

### Basic Dataset Information

- Total Entries: 4240
- Columns: 13
- Data Types: The dataset includes both integer and float types.
- Missing Values: - cigsPerDay: 29 missing - BPMeds: 53 missing - totChol: 50 missing - BMI: 19 missing - heartRate: 1 missing - glucose: 388 missing

1. Distribution of Target Variable
        - Visualization: Count plot of the Risk variable.
        - Insight:
           - The dataset is imbalanced with 68.81% low risk (0) and 31.19% high risk (1).
           - Understanding this imbalance is crucial for model training and evaluation.

2. Age Distribution
    - Visualization: Histogram of age colored by Risk with KDE.
    - Insight:
      - The age distribution shows distinct patterns for low and high-risk categories.
      - Higher age tends to have a higher risk of hypertension.

3. Gender and Risk

    - Visualization: Count plot of male variable with Risk as hue.
    - Insight:
        - Males have a slightly higher proportion of high-risk cases compared to females.

4. BMI Distribution

    - Visualization: Histogram of BMI colored by Risk with KDE and vertical lines for overweight and obese thresholds.
    - Insight:
      - High BMI is correlated with higher hypertension risk.
      - Thresholds show that many individuals in the high-risk category fall into overweight and obese ranges.

5. Blood Pressure Scatter Plot

    - Visualization: Scatter plot of sysBP vs. diaBP colored by Risk.
    - Insight:
      - There is a clear distinction between systolic and diastolic blood pressure in relation to hypertension risk.
      - High-risk individuals generally have higher systolic and diastolic blood pressures.
        
6. Correlation Heatmap

- Visualization: Heatmap of the correlation matrix of features.
- Insight:
  - Strong correlations observed between sysBP and diaBP, and between age and other features.
  - Helps in identifying multicollinearity and significant predictors for hypertension risk.

7. Box Plots for Key Numerical Features

- Visualization: Box plots of numerical features (age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose) against Risk.
- Insight:
  - These plots highlight significant differences in the distribution of numerical features between low and high-risk categories.
  - Notable differences are observed in sysBP, diaBP, and BMI.
    
8. Smoking Status and Risk

- Visualization: Count plot of currentSmoker with Risk as hue.
- Insight:
  - Current smokers are more likely to be in the high-risk category.
  - Smoking status is a significant factor in hypertension risk.

9. Diabetes and Risk

- Visualization: Count plot of diabetes with Risk as hue.
- Insight:
  - Individuals with diabetes are more likely to be at high risk of hypertension.
  - - Diabetes status is a significant predictor of hypertension risk.



# Feature Relationships:

- Strong correlation observed between systolic and diastolic blood pressure
- Age correlates with several other features
  
<img width="845" alt="Screenshot 2024-07-01 at 8 42 49 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/ed35c40a-7089-4176-a230-116fd5c88f87">

<img width="917" alt="Screenshot 2024-07-01 at 8 22 05 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/758d3f51-2876-451a-b5de-a226938d7cfc">


# Next steps

- To further enhance this project, we recommend:

1. Expanding the dataset to include more diverse populations, ensuring better model generalization
2. Exploring advanced ensemble methods or deep learning approaches for potentially higher predictive accuracy
3. Developing a user-friendly interface or API to make the model accessible to healthcare professionals
4. Conducting a longitudinal study to validate the model's predictive power over time
5. Investigating the potential for personalized risk factor identification to guide individual preventive measures
6. Incorporating additional relevant features such as family history or dietary habits
7. Collaborating with healthcare professionals to validate and refine the model's clinical applicability

# Outline of project

1. https://github.com/Ohmae22/Capstone-Project

# Contact and Further Information

https://github.com/Ohmae22
