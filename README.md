# 📉 Customer Churn Prediction

This project aims to predict customer churn using various machine learning models. The primary goal is to identify customers who are likely to stop using a service, allowing businesses to take proactive measures to retain them.

---

## 📁 Dataset

The project utilizes a customer churn dataset (`customer_churn.csv`) to train and evaluate the prediction models.

---

## 🧰 Libraries Used

The following Python libraries are used in this project:

### Data Manipulation and Analysis
- `pandas`

### Machine Learning
- `scikit-learn`:
  - `LabelEncoder`, `train_test_split`, `GridSearchCV`, `RandomizedSearchCV`
  - `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`
  - `classification_report`, `accuracy_score`, `f1_score`
  - `StandardScaler`, `OrdinalEncoder`, `SelectFromModel`
  - `XGBClassifier`, `LGBMClassifier`, `GaussianNB`, `KNeighborsClassifier`
  - `MLPClassifier`, `CatBoostClassifier`, `VotingClassifier`, `StackingClassifier`
  - `LogisticRegression`, `SVC`

- `imblearn`:
  - `SMOTE`

### Data Visualization
- `matplotlib`
- `seaborn`

---

## 🔍 Methodology

The project follows these steps:

1. **Exploratory Data Analysis (EDA)**  
   The dataset is explored to understand its structure, identify patterns, and check for class imbalance.

2. **Data Preprocessing**  
   - The `customerID` column is dropped.  
   - Missing values in `TotalCharges` (where tenure is 0) are handled by removing those rows.

3. **Model Building**  
   Various machine learning classifiers are trained on the preprocessed data, including:
   - Random Forest
   - AdaBoost
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - Gaussian Naive Bayes
   - K-Nearest Neighbors
   - Multi-layer Perceptron (MLP)
   - CatBoost
   - Voting Classifier
   - Stacking Classifier
   - Logistic Regression
   - Support Vector Classifier (SVC)

4. **Model Evaluation**  
   The models are evaluated using metrics like:
   - Accuracy
   - F1-score
   - Classification report

5. **Model Interpretability**  
   SHAP (SHapley Additive exPlanations) is used to interpret the model's predictions and understand the influence of different features on churn.

---

## 📊 Results

The project identifies key features that influence customer churn and provides insights into how these features interact.

SHAP analysis reveals that:
- Contract type
- Charges
- Payment method
- Tech support
- Online security
- Internet service
- Monthly charges  
are significant predictors of churn.

---

## 🚀 How to Use

To use this project:

1. Clone the repository.
2. Install the required libraries
3. Run the `Customer_churn_Prediction.ipynb` notebook in a Jupyter environment.

---

## 📝 Project Summary

This README provides an overview of the customer churn prediction project, including its objectives, dataset, methodologies, and key findings.
