# ============================================================
# DEEP LEARNING ASSIGNMENT
# Employee Attrition Prediction using MLPClassifier
# ============================================================


# ============================================================
# 1. LOAD THE DATASET USING PANDAS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Load dataset
df = pd.read_csv("Employee_Attrition.csv")


# ============================================================
# 2. DISPLAY THE SHAPE, COLUMNS AND FIRST 5 RECORDS
# ============================================================

print("\n============================================================")
print("DATASET INFORMATION")
print("============================================================")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())


# ============================================================
# 3. CHECK FOR MISSING VALUES
# ============================================================

print("\n============================================================")
print("MISSING VALUES")
print("============================================================")

print(df.isnull().sum())


# ============================================================
# 4. IDENTIFY CATEGORICAL FEATURES
# ============================================================

print("\n============================================================")
print("CATEGORICAL FEATURES")
print("============================================================")

categorical_columns = df.select_dtypes(include="object").columns

print(categorical_columns)


# ============================================================
# 5. CONVERT CATEGORICAL FEATURES SUCH AS OverTime
#    INTO NUMERICAL REPRESENTATION
# ============================================================

print("\n============================================================")
print("ENCODING CATEGORICAL FEATURES")
print("============================================================")

# Convert OverTime into numerical values
df["OverTime"] = df["OverTime"].map({
    "No": 0,
    "Yes": 1
})

print("\nAfter Encoding OverTime:")
print(df.head())


# ============================================================
# 6. CONVERT THE TARGET Attrition INTO 0 AND 1
# ============================================================

print("\n============================================================")
print("ENCODING TARGET VARIABLE")
print("============================================================")

df["Attrition"] = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})

print("\nTarget values after encoding:")
print(df["Attrition"].value_counts())


# ============================================================
# 7. SEPARATE INDEPENDENT AND DEPENDENT VARIABLES
# ============================================================

print("\n============================================================")
print("SEPARATING X AND Y")
print("============================================================")

# Independent variables
X = df.drop("Attrition", axis=1)

# Dependent / target variable
y = df["Attrition"]

print("\nFeatures X:")
print(X.head())

print("\nTarget y:")
print(y.head())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)


# ============================================================
# 8. DIVIDE DATA INTO TRAINING AND TESTING DATA
# ============================================================

print("\n============================================================")
print("TRAIN TEST SPLIT")
print("============================================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# 9. APPLY APPROPRIATE FEATURE SCALING
# ============================================================

print("\n============================================================")
print("FEATURE SCALING")
print("============================================================")

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")


# ============================================================
# 10. DESIGN AN MLP WITH AT LEAST TWO HIDDEN LAYERS
# ============================================================

print("\n============================================================")
print("MLP MODEL")
print("============================================================")

mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

print("\nMLP Model:")
print(mlp)


# ============================================================
# 11. TRAIN THE NETWORK
# ============================================================

print("\n============================================================")
print("MODEL TRAINING")
print("============================================================")

mlp.fit(X_train_scaled, y_train)

print("\nModel training completed successfully.")


# ============================================================
# 12. DISPLAY THE NUMBER OF ITERATIONS REQUIRED FOR TRAINING
# ============================================================

print("\n============================================================")
print("NUMBER OF ITERATIONS")
print("============================================================")

print("\nNumber of iterations required:")
print(mlp.n_iter_)


# ============================================================
# 13. CALCULATE TRAINING ACCURACY
# ============================================================

print("\n============================================================")
print("TRAINING ACCURACY")
print("============================================================")

y_train_pred = mlp.predict(X_train_scaled)

training_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print("\nTraining Accuracy:")
print(training_accuracy)

print("\nTraining Accuracy Percentage:")
print(training_accuracy * 100)


# ============================================================
# TEST DATA PREDICTION
# ============================================================

y_test_pred = mlp.predict(X_test_scaled)


# ============================================================
# TESTING ACCURACY
# ============================================================

testing_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("\nTesting Accuracy:")
print(testing_accuracy)

print("\nTesting Accuracy Percentage:")
print(testing_accuracy * 100)


# ============================================================
# 14. CALCULATE TESTING ACCURACY
# ============================================================

print("\n============================================================")
print("TESTING ACCURACY")
print("============================================================")

print("Testing Accuracy:", testing_accuracy)
print("Testing Accuracy Percentage:", testing_accuracy * 100)


# ============================================================
# 15. GENERATE A CONFUSION MATRIX
# ============================================================

print("\n============================================================")
print("CONFUSION MATRIX")
print("============================================================")

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\nConfusion Matrix:")
print(cm)


# Plot confusion matrix
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Employee Attrition - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["No Attrition", "Attrition"]
)

plt.yticks(
    [0, 1],
    ["No Attrition", "Attrition"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.show()


# ============================================================
# 16. PLOT THE LOSS CURVE
# ============================================================

print("\n============================================================")
print("LOSS CURVE")
print("============================================================")

plt.figure(figsize=(8, 5))

plt.plot(
    mlp.loss_curve_
)

plt.title("MLP Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.grid()

plt.show()


# ============================================================
# 17. CREATE A FUNCTION PredictAttrition(employee_data)
# ============================================================

print("\n============================================================")
print("PREDICTION FUNCTION")
print("============================================================")


def PredictAttrition(employee_data):

    # Convert input into DataFrame
    employee_df = pd.DataFrame([employee_data])

    # Convert OverTime into numerical value
    employee_df["OverTime"] = employee_df["OverTime"].map({
        "No": 0,
        "Yes": 1
    })

    # Scale input data
    employee_scaled = scaler.transform(employee_df)

    # Make prediction
    prediction = mlp.predict(employee_scaled)

    # Get probability
    probability = mlp.predict_proba(employee_scaled)

    if prediction[0] == 0:
        result = "Employee is likely to stay"
    else:
        result = "Employee is likely to leave"

    print("\nPrediction:", result)

    print(
        "Probability of Leaving:",
        probability[0][1]
    )

    return result


# ============================================================
# 18. TEST THE SYSTEM USING AT LEAST FIVE NEW EMPLOYEE RECORDS
# ============================================================

print("\n============================================================")
print("TESTING WITH 5 NEW EMPLOYEES")
print("============================================================")


employee_1 = {
    "Age": 25,
    "MonthlyIncome": 30000,
    "YearsAtCompany": 2,
    "TotalWorkingYears": 3,
    "DistanceFromHome": 5,
    "JobSatisfaction": 4,
    "WorkLifeBalance": 4,
    "OverTime": "No",
    "NumCompaniesWorked": 1,
    "TrainingTimesLastYear": 3
}


employee_2 = {
    "Age": 45,
    "MonthlyIncome": 80000,
    "YearsAtCompany": 10,
    "TotalWorkingYears": 20,
    "DistanceFromHome": 10,
    "JobSatisfaction": 3,
    "WorkLifeBalance": 3,
    "OverTime": "Yes",
    "NumCompaniesWorked": 3,
    "TrainingTimesLastYear": 2
}


employee_3 = {
    "Age": 30,
    "MonthlyIncome": 45000,
    "YearsAtCompany": 4,
    "TotalWorkingYears": 7,
    "DistanceFromHome": 15,
    "JobSatisfaction": 2,
    "WorkLifeBalance": 2,
    "OverTime": "Yes",
    "NumCompaniesWorked": 2,
    "TrainingTimesLastYear": 1
}


employee_4 = {
    "Age": 50,
    "MonthlyIncome": 100000,
    "YearsAtCompany": 15,
    "TotalWorkingYears": 25,
    "DistanceFromHome": 3,
    "JobSatisfaction": 4,
    "WorkLifeBalance": 4,
    "OverTime": "No",
    "NumCompaniesWorked": 2,
    "TrainingTimesLastYear": 4
}


employee_5 = {
    "Age": 28,
    "MonthlyIncome": 35000,
    "YearsAtCompany": 1,
    "TotalWorkingYears": 4,
    "DistanceFromHome": 20,
    "JobSatisfaction": 1,
    "WorkLifeBalance": 2,
    "OverTime": "Yes",
    "NumCompaniesWorked": 4,
    "TrainingTimesLastYear": 1
}


# Make predictions
print("\nEmployee 1:")
PredictAttrition(employee_1)

print("\nEmployee 2:")
PredictAttrition(employee_2)

print("\nEmployee 3:")
PredictAttrition(employee_3)

print("\nEmployee 4:")
PredictAttrition(employee_4)

print("\nEmployee 5:")
PredictAttrition(employee_5)


# ============================================================
# 19. EXPLAIN OVERFITTING AND UNDERFITTING
# ============================================================

print("\n============================================================")
print("OVERFITTING AND UNDERFITTING")
print("============================================================")

print("""
OVERFITTING:
Overfitting occurs when the model learns the training data
too well, including noise and unnecessary patterns.

If training accuracy is very high but testing accuracy is
much lower, the model may be overfitting.

Ways to reduce overfitting:
1. Reduce model complexity.
2. Use regularization.
3. Use more training data.
4. Use early stopping.


UNDERFITTING:
Underfitting occurs when the model is too simple to learn
the important patterns in the data.

If both training and testing accuracy are low, the model
may be underfitting.

Ways to reduce underfitting:
1. Increase model complexity.
2. Increase hidden layers or neurons.
3. Train the model for more iterations.
4. Improve the input features.
""")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("FINAL MODEL SUMMARY")
print("============================================================")

print("Training Accuracy:", training_accuracy * 100, "%")
print("Testing Accuracy :", testing_accuracy * 100, "%")
print("Iterations        :", mlp.n_iter_)

print("\nEmployee Attrition Prediction System Completed!")