# Industrial Machine Failure Prediction
## 1. Introduction
Unexpected industrial machine failures can cause downtime, repair costs, and production delays. Predictive maintenance uses sensor measurements and machine-learning models to estimate the likelihood of failure before it occurs.

## 2. Objectives
- Analyse machine operating conditions and sensor measurements.
- Predict the binary target: Machine Failure (Yes/No).
- Train Logistic Regression, Decision Tree, and Random Forest classifiers.
- Compare accuracy, precision, recall, F1-score, and confusion matrices.
- Apply five-fold cross-validation and GridSearchCV.
- Save the best pipeline and predict unseen machine records.

## 3. Dataset
The recommended dataset is the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository. It contains 10,000 records. Important features include machine type, air temperature, process temperature, rotational speed, torque, and tool wear. The included CSV is synthetic demonstration data only. For formal analysis, use the official dataset and retain its documentation.

## 4. Data preparation
The target is `Machine failure`. Identifier fields are excluded. Failure-mode indicators such as TWF, HDF, PWF, OSF, and RNF are excluded from predictors because they are target-related and can cause data leakage. Numerical features are imputed and scaled; categorical features are imputed and one-hot encoded.

## 5. Methodology
1. Load and inspect the CSV with Pandas.
2. Check missing values, duplicates, class distribution, and feature summaries.
3. Split the data into 80% training and 20% test data using stratification.
4. Train Logistic Regression, Decision Tree, and Random Forest classifiers.
5. Evaluate with accuracy, precision, recall, F1-score, and confusion matrix.
6. Use five-fold stratified cross-validation.
7. Tune model parameters with GridSearchCV using F1-score.
8. Select the best cross-validation model and evaluate on the held-out test set.
9. Save the full preprocessing and model pipeline with Joblib.

## 6. Expected output
A model comparison, cross-validation scores, selected hyperparameters, confusion matrix, saved model, and a prediction of Machine Failure: Yes/No for new machine measurements.

## 7. Limitations
The synthetic sample is for demonstrating the project structure and code only. It cannot support industrial conclusions. Real deployment requires the official dataset or validated plant data, domain-expert review, monitoring, and safety procedures. A model prediction must not replace maintenance inspection.

## 8. Conclusion
The project demonstrates a predictive-maintenance classification workflow. Future improvements may include real-time sensor integration, drift monitoring, explainability, and cost-sensitive threshold selection.
