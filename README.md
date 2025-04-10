# Workforce Attrition Predictor 

A machine learning model to predict employee attrition using HR dataset features like age, salary, overtime, and work experience. Built using Logistic Regression + GridSearchCV for hyperparameter tuning.

## Model Performance
- Accuracy: **86.3%**
- Model: Logistic Regression with ElasticNet penalty
- Feature Engineering: Removed unnecessary features like `MaritalStatus`, applied binary encoding to categorical columns

## Tech Stack
- Python 
- Pandas, Scikit-Learn
- Logistic Regression
- Pipeline + GridSearchCV

## Key Learnings
- Feature selection improves performance significantly
- Regularization may underfit if feature space is already optimized
- Logistic Regression works great for balanced HR attrition datasets

## Future Scope
- Try Random Forests and XGBoost
- Visualize feature importances
- Build a Streamlit dashboard

## Files
- `project.py` – Final Python script
- `requirements.txt` – Python dependencies
