import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
data=pd.read_csv('workforce.csv')
df=pd.DataFrame(data)

selected_cols = [
    'Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'Gender', 'HourlyRate', 'JobLevel', 'JobRole',
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
    'PercentSalaryHike', 'PerformanceRating', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'JobSatisfaction'
]
numeric_cols = df[selected_cols].select_dtypes(include='number').columns.tolist()
df1 = df.groupby('Department')[numeric_cols].mean()
# since there are no null values we proceed
df = df[selected_cols]
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Step 2: Drop rows with nulls if needed (or handle them differently)
df = df.dropna()

# Step 3: Define X and y
x = df[numeric_cols]
y = df['Attrition']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


pipe=Pipeline([
    ('scaler',StandardScaler()),
    ('classification',LogisticRegression())
])

param_grid={
    'classification__C':[0.5,0.8,1],
    'classification__penalty':['elasticnet'],
    'classification__solver':['saga'],
    'classification__l1_ratio':[0.2,0.5,0.8]
}

grid=GridSearchCV(pipe,cv=5,n_jobs=-1,verbose=0,scoring='accuracy',param_grid=param_grid)

grid.fit(x_train,y_train)

best_model=grid.best_estimator_
y_pred=best_model.predict(x_test)
accuracy=accuracy_score(y_true=y_test,y_pred=y_pred)
report=classification_report(y_true=y_test,y_pred=y_pred)
print(f'The accuracy of this model is: {accuracy}')
print(f'The classification report of this model is {report}')

