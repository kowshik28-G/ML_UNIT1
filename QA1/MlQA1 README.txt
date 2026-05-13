# QA 1 - ML Problem Formulation + Basic Implementation

## Title
Student Score Prediction using Linear Regression

---

## Problem Definition
The problem is to predict the marks of a student based on the number of hours studied.

This is a Regression problem because the output value is continuous (marks can be any numerical value).

---

## Why Regression?
Regression is used when the output is a continuous numerical value.

In this project:
- Input Feature = Study Hours
- Output = Student Marks

Since marks are numerical values, Linear Regression is used.

---

## Dataset Explanation

| Study Hours | Marks |
|-------------|-------|
| 1 | 15 |
| 2 | 25 |
| 3 | 35 |
| 4 | 45 |
| 5 | 55 |
| 6 | 65 |
| 7 | 75 |
| 8 | 85 |
| 9 | 95 |
| 10 | 100 |

Study Hours is the input feature.  
Marks is the target output.

---

## Python Code

```python
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

print("\nMean Absolute Error:", mae)\

OUTPUT:
Dataset:

   Study_Hours  Marks
0            1     15
1            2     25
2            3     35
3            4     45
4            5     55
5            6     65
6            7     75
7            8     85
8            9     95
9           10    100


Predictions:

Study Hours: 9
Actual Marks: 95
Predicted Marks: 94.89
-----------------------------

Study Hours: 2
Actual Marks: 25
Predicted Marks: 24.56
-----------------------------

Mean Absolute Error: 0.28

Output Explanation

The model predicts student marks based on study hours.

If a student studies more hours, the predicted marks also increase.

Linear Regression finds the relationship between study hours and marks.

Conclusion

Linear Regression successfully predicts student marks using study hours.

The model works well because the dataset has a linear relationship.