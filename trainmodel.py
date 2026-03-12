import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r'C:\Users\HP\OneDrive\Documents\PROJECTS\DATA SCIENCE PROJECT\IRIS\Iris.csv')

# Encode target variable
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

# Features and target
X = df.drop(['Id','Species'], axis=1)
y = df['Species']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and evaluate
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f'{name} Accuracy: {accuracy}')

# Hyperparameter tuning
param_grid = {
    'n_estimators':[50,100,200],
    'max_depth':[None,5,10],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
    'max_features':['sqrt','log2']
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Best Random Forest Accuracy:", accuracy)

# Save trained model
joblib.dump(best_model, "iris_model.pkl")

print("Model saved successfully")