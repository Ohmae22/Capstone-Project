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


# Methodology
- Our approach followed these key steps:

## 1. Data Preprocessing:

    - Handled missing values in several columns (cigsPerDay, BPMeds, totChol, BMI, heartRate, glucose)
    - Encoded categorical variables
    - Scaled numerical features


## 2. Exploratory Data Analysis (EDA):

    - Analyzed the distribution of the target variable (Risk)
    - Explored relationships between features and hypertension risk
    - Created visualizations to understand feature interactions and distributions


## 3. Feature Selection and Engineering:

    - Applied Principal Component Analysis (PCA) for dimensionality reduction
    - Utilized Sequential Feature Selection (SFS) to identify optimal feature subsets


## 4. Model Development:

    - Split data into training and testing sets
    - Implemented multiple models:

      a) K-Mans Clustering
      b) Linear Regression
      c) Ridge Regression
      d) Lasso Regression
    
## 5. Model Evaluation:

    - Utilized cross-validation to ensure robust performance estimates
    - Assessed models using appropriate metrics (e.g., MSE, RMSE for regression tasks)
    - Compared the performance of different techniques


## 6. Hyperparameter Tuning & Ensemble technique:

    - Performed GridSearchCV on the best-performing model to optimize its parameters
    - Optimisation with Ensemble technique with Randon Forest


## 7. Interpretation and Insights:

    - Analyzed feature importance
    - Interpreted model results in the context of hypertension risk prediction

# Results

- Our analysis yielded the following key findings:

## 1. Model Performance:

<img width="342" alt="Screenshot 2024-07-01 at 9 49 04 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/ea25c5d9-9596-45f5-96e5-2d0a2e1a5239">

  - Key Observations:
    
    - Improvement with Feature Engineering:
      - a) Both Linear Regression and Ridge Regression models showed improvement in performance when polynomial features were added. This is evidenced by a decrease in MSE and an increase in R^2 values.
      - b) Specifically, the MSE for Linear Regression dropped from 0.489975 to 0.475583, and the R^2 value increased from 0.498800 to 0.513522.
      - c) Similarly, Ridge Regression saw a decrease in MSE from 0.489973 to 0.475056, and the R^2 value increased from 0.498802 to 0.514061.

    - Lasso Regression Performance:
      - a) The Lasso Regression model did not show any improvement with the addition of polynomial features. The MSE and R^2 values remained the same, indicating that Lasso Regression did not benefit from the feature engineering in this context.
  
    - Best Performing Model:
      - a) The best performing model overall is the Ridge Regression with Polynomial Features, with the lowest MSE of 0.475056 and the highest R^2 value of 0.514061. This suggests that Ridge Regression with polynomial features captures the underlying patterns in the data more effectively compared to other models.

    - Feature Engineering Effectiveness:
      - a) The use of polynomial features has proven to be effective in improving the performance of Linear and Ridge Regression models, but not for Lasso Regression. This indicates that feature engineering can be highly beneficial but its impact can vary depending on the model used.
     
    - Optimisation with Ensemble Technique - Random Forest
        - a) The Random Forest model outperforms all other models, achieving the highest R^2 score of 0.640824. This indicates that the Random Forest explains about 64% of the variance in the target variable (hypertension risk).
        - b) The Voting Ensemble comes in second with an R^2 score of 0.575249, suggesting that combining multiple models does improve performance over individual models, but not as much as the Random Forest alone.
          

    <img width="594" alt="Screenshot 2024-07-27 at 2 15 58 PM" src="https://github.com/user-attachments/assets/69d52b76-0808-425d-985a-bc8ddf0dec3d">
     

      <img width="1039" alt="Screenshot 2024-07-27 at 2 16 19 PM" src="https://github.com/user-attachments/assets/43b18a98-cb5d-49db-9ef2-397de07e8bf3">

      

## 2. Key Predictive Features:

  a) Systolic Blood Pressure (sysBP)
  b) Age
  c) Body Mass Index (BMI)
  d) Diastolic Blood Pressure (diaBP)
  e) Total Cholesterol (totChol)

## 3. Key Findings:

  a) Best Prediction Method: We found that a technique called "Ridge Regression with Polynomial Features" worked best for predicting hypertension risk. This method was able to correctly identify risk levels about
    51% of the time, which is a significant improvement over random guessing.
    
  b) However further optimisation with ensemble technique like Randon Forest further improved the prediction accuracy up to ~64% of R^2 score. Given the Random Forest's superior performance, it should be the primary model used for predictions. However, the trade-off between performance and interpretability should be considered. While Random Forest performs best, it's less interpretable than linear models. Focus on accurately measuring and monitoring systolic blood pressure as it's the most crucial predictor. Consider developing a two-stage model: first identifying high-risk individuals based on easily obtainable information, then recommending more detailed testing for those individuals.

  c) Most Important Risk Factors: The study identified several key factors that strongly influence hypertension risk:
      - Systolic blood pressure (the top number in a blood pressure reading)
      - Age
      - Body Mass Index (BMI)
      - Diastolic blood pressure (the bottom number in a blood pressure reading)
      - Total cholesterol levels

  d) Age and Gender Insights:
      - Older individuals were more likely to be at high risk for hypertension.
      - Men showed a slightly higher proportion of high-risk cases compared to women.

  e) Lifestyle Factors:
      - People with higher BMIs were more likely to be at risk for hypertension.
      - Current smokers had a higher chance of being in the high-risk category.
      - Individuals with diabetes were more likely to be at high risk for hypertension.

## Implications:

  a) Early Intervention: This tool could help doctors identify patients who might develop hypertension before they show symptoms. This early warning could lead to preventive measures like lifestyle changes or early treatment.

  b) Personalized Health Plans: By knowing which factors are most important, healthcare providers can create more targeted health plans. For example, they might focus more on weight management for patients with high BMIs.

  c) Public Health Campaigns: The findings could inform public health initiatives. For instance, anti-smoking campaigns could emphasize the link between smoking and hypertension risk.

  d) Resource Allocation: Healthcare systems could use this information to allocate resources more effectively, focusing on high-risk individuals and the most impactful risk factors.


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

## 1. Distribution of Target Variable
        - Visualization: Count plot of the Risk variable.
        - Insight:
           - The dataset is imbalanced with 68.81% low risk (0) and 31.19% high risk (1).
           - Understanding this imbalance is crucial for model training and evaluation.

## 2. Age Distribution
    - Visualization: Histogram of age colored by Risk with KDE.
    - Insight:
      - The age distribution shows distinct patterns for low and high-risk categories.
      - Higher age tends to have a higher risk of hypertension.


  <img width="774" alt="Screenshot 2024-07-01 at 8 37 32 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/55878198-1bbe-4d12-a4ea-61084034ceab">

## 3. Gender and Risk

    - Visualization: Count plot of male variable with Risk as hue.
    - Insight:
        - Males have a slightly higher proportion of high-risk cases compared to females.

    <img width="706" alt="Screenshot 2024-07-01 at 8 41 33 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/03b1ba4e-2bfa-4614-8bfa-20e3305bc4d0">

## 4. BMI Distribution

    - Visualization: Histogram of BMI colored by Risk with KDE and vertical lines for overweight and obese thresholds.
    - Insight:
      - High BMI is correlated with higher hypertension risk.
      - Thresholds show that many individuals in the high-risk category fall into overweight and obese ranges.

<img width="792" alt="Screenshot 2024-07-01 at 9 33 53 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/54d1d28d-2f32-4992-8072-ca65aa8f6d6f">

## 5. Blood Pressure Scatter Plot

    - Visualization: Scatter plot of sysBP vs. diaBP colored by Risk.
    - Insight:
      - There is a clear distinction between systolic and diastolic blood pressure in relation to hypertension risk.
      - High-risk individuals generally have higher systolic and diastolic blood pressures.
     
  <img width="845" alt="Screenshot 2024-07-01 at 8 42 49 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/ed35c40a-7089-4176-a230-116fd5c88f87">
        
## 6. Correlation Heatmap

- Visualization: Heatmap of the correlation matrix of features.
- Insight:
  - Strong correlations observed between sysBP and diaBP, and between age and other features.
  - Helps in identifying multicollinearity and significant predictors for hypertension risk.

<img width="917" alt="Screenshot 2024-07-01 at 8 22 05 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/758d3f51-2876-451a-b5de-a226938d7cfc">

## 7. Box Plots for Key Numerical Features

- Visualization: Box plots of numerical features (age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose) against Risk.
- Insight:
  - These plots highlight significant differences in the distribution of numerical features between low and high-risk categories.
  - Notable differences are observed in sysBP, diaBP, and BMI.
    
## 8. Smoking Status and Risk

- Visualization: Count plot of currentSmoker with Risk as hue.
- Insight:
  - Current smokers are more likely to be in the high-risk category.
  - Smoking status is a significant factor in hypertension risk.

## 9. Diabetes and Risk

- Visualization: Count plot of diabetes with Risk as hue.
- Insight:
  - Individuals with diabetes are more likely to be at high risk of hypertension.
  - - Diabetes status is a significant predictor of hypertension risk.



# Feature Relationships:

- Strong correlation observed between systolic and diastolic blood pressure
- Age correlates with several other features
- Feature importance from Linear Regression, Ridge Regression, and Lasso Regression models.
  
<img width="776" alt="Screenshot 2024-07-01 at 9 27 34 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/c6bce7e7-a714-45e6-93f4-afa5e46f8f7e">

<img width="742" alt="Screenshot 2024-07-01 at 9 27 40 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/599b45c1-e4a5-4a82-8a48-c174c096023d">

<img width="749" alt="Screenshot 2024-07-01 at 9 27 46 PM" src="https://github.com/Ohmae22/Capstone-Project/assets/88304497/3cd8d0b9-7a78-4244-ae85-028058a916f4">


# Next steps

- To further enhance this project, we recommend:

1. Conduct feature engineering to compare initial result
2. Expanding the dataset to include more diverse populations, ensuring better model generalization
3. Exploring advanced ensemble methods or deep learning approaches for potentially higher predictive accuracy
4. Developing a user-friendly interface or API to make the model accessible to healthcare professionals
5. Conducting a longitudinal study to validate the model's predictive power over time
6. Investigating the potential for personalized risk factor identification to guide individual preventive measures
7. Incorporating additional relevant features such as family history or dietary habits
8. Collaborating with healthcare professionals to validate and refine the model's clinical applicability



# Outline of project

1. https://github.com/Ohmae22/Capstone-Project/blob/main/EDA.ipynb
2. https://github.com/Ohmae22/Capstone-Project/blob/main/Modelling.ipynb

# Contact and Further Information

https://github.com/Ohmae22
