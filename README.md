# Depression Prediction Using Machine Learning  

## Overview  
The **Depression Prediction** project aims to use machine learning algorithms to predict whether an individual is likely to experience depression. The goal is to analyze multiple factors such as work/study pressure, family history, sleep patterns, and lifestyle to provide insights into what leads to mental health challenges and predict depressive tendencies. This project offers a predictive model for early intervention and mental health support.

## Table of Contents  
- [Project Overview](#overview)  
- [Technologies Used](#technologies-used)  
- [Dataset](#dataset)  
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Data Preprocessing](#data-preprocessing)  
- [Machine Learning Models](#machine-learning-models)  
- [Evaluation](#evaluation)  
- [Results](#results)  


## Technologies Used  
This project was developed using the following technologies:

- **Programming Language**: Python  
- **Libraries**:  
  - **Data Manipulation & Analysis**: Pandas, NumPy  
  - **Data Visualization**: Matplotlib, Seaborn  
  - **Machine Learning**: Scikit-learn (for model building, preprocessing, and evaluation)  
  - **Jupyter Notebooks**: For interactive data analysis and model evaluation  

## Dataset  
The dataset used in this project contains information about individuals, such as:

- **Demographics**: `id`, `Name`, `Gender`, `Age`, `City`  
- **Work/Study Environment**: `Working Professional or Student`, `Profession`, `Academic Pressure`, `Work Pressure`, `Study Satisfaction`, `Job Satisfaction`  
- **Health & Lifestyle**: `Sleep Duration`, `Dietary Habits`, `Degree`, `Work/Study Hours`, `Financial Stress`  
- **Mental Health Factors**: `Family History of Mental Illness`, `Have you ever had suicidal thoughts?`, `Depression` (Target Variable)  

**Target Variable**:  
- `Depression`: The label indicating whether an individual is diagnosed with depression (`Yes/No`).

### Data Collection
The data was sourced from a survey capturing self-reported mental health indicators and lifestyle data. It reflects various demographic, work-related, and health aspects that can contribute to depression.

## Exploratory Data Analysis (EDA)  
Exploratory Data Analysis (EDA) was performed to understand the structure of the data and identify patterns in mental health:

- **Data Distribution**:  
  Visualized the distribution of continuous features like `Age`, `CGPA`, and `Sleep Duration` using histograms and boxplots.
  
- **Correlation Analysis**:  
  Used heatmaps and pair plots to investigate correlations between features, such as how `Work Pressure` and `Study Satisfaction` relate to the target variable `Depression`.

- **Missing Data & Outliers**:  
  - Visualized missing data using missing data heatmaps.  
  - Identified outliers in features such as `Age`, `Work Pressure`, and `Sleep Duration` that could affect model performance.

## Data Preprocessing  
Data preprocessing involved several steps to prepare the data for machine learning:

1. **Handling Missing Values**:  
   Missing values were handled using imputation methods such as mean or median imputation, depending on the feature type.

2. **Encoding Categorical Variables**:  
   - Categorical columns like `Gender`, `Profession`, and `Degree` were encoded using **LabelEncoder** and **OneHotEncoder** to convert text labels into numeric representations.

3. **Feature Scaling**:  
   Continuous features like `Age`, `CGPA`, `Work/Study Hours` were scaled using **StandardScaler** to ensure all variables were on the same scale.

4. **Feature Transformation**:  
   Transformed features that were highly skewed (e.g., `CGPA`) using **Log Transformation** to normalize their distribution.

5. **Data Splitting**:  
   The dataset was split into training and test sets (80/20 split) to ensure that the model was evaluated on unseen data.

## Machine Learning Models  
Several machine learning models were trained to predict depression based on the available features:

1. **Logistic Regression**:  
   A simple linear model was used to predict the binary outcome (Depression or Not). It helps in understanding the relationship between features and the target.

2. **Random Forest Classifier**:  
   A more complex ensemble model that can capture non-linear relationships and interactions between features. It was also used to determine feature importance.

3. **Support Vector Machine (SVM)**:  
   A classification model that works well in high-dimensional spaces. Used for capturing complex boundaries in the feature space.

4. **K-Nearest Neighbors (KNN)**:  
   A non-parametric model that classifies based on proximity to other data points.

5. ** Light Gradient Boosting Machines (LGBM)**:  
   Another ensemble model, focusing on combining the predictions of multiple weak learners (typically decision trees) to improve accuracy.

## Evaluation  
The models were evaluated based on various classification metrics:

- **Accuracy**: The percentage of correct predictions.  
- **Precision**: The ability of the model to correctly predict the positive class (depression).  
- **Recall**: The ability to identify actual positive cases (depressed individuals).  
- **F1-Score**: The harmonic mean of precision and recall, used to evaluate models where both false positives and false negatives are important.  
- **Confusion Matrix**: Used to visualize the true positive, true negative, false positive, and false negative predictions.  

## Results  
The best-performing model was selected based on the F1-Score and accuracy. Key findings include:

- **Feature Importance**:  
  - `Family History of Mental Illness`, `Work Pressure`, Academic Pressure and `Sleep Duration` were found to be the most significant features contributing to depression.
  
- **Model Performance**:  
  - **Random Forest** and **Gradient Boosting** models outperformed other algorithms, achieving high accuracy and recall.  
  - The **Logistic Regression** model provided interpretable insights but performed slightly worse in predicting depression compared to ensemble methods.

## How to Run  

1. **Clone the Repository**:  
   Clone the project repository to your local machine.  
   ```bash  
   git clone https://github.com/your-username/depression-prediction.git  
