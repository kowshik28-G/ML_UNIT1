# QA 1 - Student Score Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Create Dataset

data = {
    'Study_Hours': [1,2,3,4,5,6,7,8,9,10],
    'Marks': [15,25,35,45,55,65,75,85,95,100]
}

# Create DataFrame

df = pd.DataFrame(data)

print("Dataset:\n")
print(df)

# Features and Target

X = df[['Study_Hours']]
y = df['Marks']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Linear Regression Model

model = LinearRegression()

model.fit(X_train, y_train)

# Predictions

predictions = model.predict(X_test)

print("\nPredictions:\n")

for i in range(len(X_test)):

    print("Study Hours:", X_test.iloc[i,0])
    print("Actual Marks:", y_test.iloc[i])
    print("Predicted Marks:", round(predictions[i],2))
    print("-----------------------------")

# Mean Absolute Error

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", mae)
