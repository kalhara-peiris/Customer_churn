# Customer_churn
Customer Churn Prediction
This project aims to predict customer churn using various machine learning models. The primary goal is to identify customers who are likely to stop using a service, allowing businesses to take proactive measures to retain them.

Dataset
The project utilizes a customer churn dataset (customer_churn.csv) to train and evaluate the prediction models.

Libraries Used
The following Python libraries are used in this project:

Data Manipulation and Analysis: pandas
Machine Learning: scikit-learn (LabelEncoder, train_test_split, GridSearchCV, RandomizedSearchCV, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, classification_report, accuracy_score, f1_score, StandardScaler, OrdinalEncoder, SelectFromModel, XGBClassifier, LGBMClassifier, GaussianNB, KNeighborsClassifier, MLPClassifier, CatBoostClassifier, VotingClassifier, StackingClassifier, LogisticRegression, SVC), imblearn (SMOTE)
Data Visualization: matplotlib, seaborn
Methodology
The project follows these steps:

Exploratory Data Analysis (EDA): The dataset is explored to understand its structure, identify patterns, and check for class imbalance.
Data Preprocessing: The 'customerID' column is dropped, and missing values in 'TotalCharges' (where tenure is 0) are handled by removing those rows.
Model Building: Various machine learning classifiers are trained on the preprocessed data, including:
Random Forest
AdaBoost
Gradient Boosting
XGBoost
LightGBM
Gaussian Naive Bayes
K-Nearest Neighbors
Multi-layer Perceptron (MLP)
CatBoost
Voting Classifier
Stacking Classifier
Logistic Regression
Support Vector Classifier (SVC)
Model Evaluation: The models are evaluated using metrics like accuracy and F1-score, and a classification report is generated.
Model Interpretability: SHAP (SHapley Additive exPlanations) is used to interpret the model's predictions and understand the influence of different features on churn.
Results
The project identifies key features that influence customer churn and provides insights into how these features interact. The SHAP analysis reveals that contract type, charges, payment method, tech support, online security, internet service, and monthly charges are significant predictors of churn.

How to Use
To use this project:

Clone the repository.
Install the required libraries 
Run the Customer_churn_Prediction.ipynb notebook in a Jupyter environment.
