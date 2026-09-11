# ============================================================
# Loan Default Prediction using Multi-Layer Perceptron
# ============================================================

# ============================================================
# 1. LOAD AND UNDERSTAND THE DATASET
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Loan_Default.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nColumn Names:")
print(df.columns.tolist())

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 2. PERFORM EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nData Types:")
print(df.dtypes)

# Distribution of numerical features
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# 3. FIND MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 4. CHECK WHETHER THE TARGET CLASSES ARE BALANCED
# ============================================================

print("\nTarget Class Distribution:")
print(df["Default"].value_counts())

print("\nTarget Class Percentage:")
print(df["Default"].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Default")

plt.title("Distribution of Default Classes")
plt.xlabel("Default")
plt.ylabel("Count")

plt.show()


# ============================================================
# 5. ENCODE CATEGORICAL VARIABLES
# ============================================================

# Check categorical columns
print("\nCategorical Columns:")
print(df.select_dtypes(include="object").columns)

# Convert categorical variables into numerical form
df_encoded = pd.get_dummies(
    df,
    columns=["PreviousDefault", "HomeOwnership"],
    drop_first=True
)

print("\nEncoded Dataset:")
print(df_encoded.head())


# ============================================================
# 6. SEPARATE X AND Y
# ============================================================

X = df_encoded.drop("Default", axis=1)

y = df_encoded["Default"]

print("\nFeatures X:")
print(X.head())

print("\nTarget y:")
print(y.head())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)


# ============================================================
# 7. SPLIT INTO TRAINING AND TESTING DATA
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ============================================================
# 8. EXPLAIN WHETHER STRATIFIED SPLITTING SHOULD BE USED
# ============================================================

print("\nTraining Class Distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting Class Distribution:")
print(y_test.value_counts(normalize=True))


# ============================================================
# FEATURE SCALING
# ============================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 9 & 10. CREATE AN MLPCLASSIFIER
# ============================================================

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

print("\nMLP Model:")
print(mlp)


# ============================================================
# 11. TRAIN THE MODEL
# ============================================================

mlp.fit(X_train_scaled, y_train)

print("\nModel Training Completed!")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

y_pred = mlp.predict(X_test_scaled)


# ============================================================
# 12. CALCULATE ACCURACY
# ============================================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100)


# ============================================================
# 13. GENERATE THE CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low Default Risk", "High Default Risk"],
    yticklabels=["Low Default Risk", "High Default Risk"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# ============================================================
# 14. GENERATE THE CLASSIFICATION REPORT
# ============================================================

from sklearn.metrics import classification_report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low Default Risk", "High Default Risk"]
    )
)


# ============================================================
# 15. CALCULATE PRECISION, RECALL AND F1-SCORE
# ============================================================

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\nPrecision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)


# ============================================================
# 16. PLOT TRAINING LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(mlp.loss_curve_)

plt.title("MLP Training Loss Curve")

plt.xlabel("Iterations")

plt.ylabel("Loss")

plt.grid()

plt.show()


# ============================================================
# 17. TEST THE MODEL ON NEW LOAN APPLICANTS
# ============================================================

# New applicants
new_applicants = pd.DataFrame([
    
    {
        "Age": 30,
        "Income": 50000,
        "LoanAmount": 150000,
        "CreditScore": 750,
        "EmploymentYears": 5,
        "ExistingLoans": 1,
        "MonthlyDebt": 10000,
        "LoanTerm": 60,
        "PreviousDefault": "No",
        "HomeOwnership": "Rent"
    },
    
    {
        "Age": 45,
        "Income": 30000,
        "LoanAmount": 500000,
        "CreditScore": 550,
        "EmploymentYears": 2,
        "ExistingLoans": 3,
        "MonthlyDebt": 25000,
        "LoanTerm": 120,
        "PreviousDefault": "Yes",
        "HomeOwnership": "Rent"
    }
])

# Encode new applicants
new_applicants_encoded = pd.get_dummies(
    new_applicants,
    columns=["PreviousDefault", "HomeOwnership"],
    drop_first=True
)

# Ensure same columns as training data
new_applicants_encoded = new_applicants_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

# Scale new data
new_applicants_scaled = scaler.transform(
    new_applicants_encoded
)

# Prediction
new_predictions = mlp.predict(
    new_applicants_scaled
)

# Probability
new_probabilities = mlp.predict_proba(
    new_applicants_scaled
)

print("\nNEW APPLICANT PREDICTIONS")

for i in range(len(new_predictions)):
    
    print("\nApplicant", i + 1)
    
    if new_predictions[i] == 0:
        print("Prediction: Low Default Risk")
    else:
        print("Prediction: High Default Risk")
    
    print(
        "Probability of Default:",
        new_probabilities[i][1]
    )


# ============================================================
# HYPERPARAMETER EXPERIMENT
# Change one parameter at a time
# ============================================================


# ============================================================
# EXPERIMENT 1 - ACTIVATION FUNCTION
# ============================================================

activations = [
    "identity",
    "logistic",
    "tanh",
    "relu"
]

activation_results = []

for activation in activations:
    
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation=activation,
        solver="adam",
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, predictions)
    
    activation_results.append({
        "Activation": activation,
        "Accuracy": acc
    })


activation_results_df = pd.DataFrame(
    activation_results
)

print("\nEXPERIMENT 1: ACTIVATION FUNCTION")
print(activation_results_df)


# ============================================================
# EXPERIMENT 2 - HIDDEN LAYERS
# ============================================================

hidden_layer_options = [
    (10,),
    (20, 10),
    (50, 25),
    (100, 50, 25)
]

hidden_layer_results = []

for layers in hidden_layer_options:
    
    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, predictions)
    
    hidden_layer_results.append({
        "Hidden Layers": str(layers),
        "Accuracy": acc
    })


hidden_layer_results_df = pd.DataFrame(
    hidden_layer_results
)

print("\nEXPERIMENT 2: HIDDEN LAYERS")
print(hidden_layer_results_df)


# ============================================================
# EXPERIMENT 3 - LEARNING RATE
# ============================================================

learning_rates = [
    0.0001,
    0.001,
    0.01
]

learning_rate_results = []

for lr in learning_rates:
    
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, predictions)
    
    learning_rate_results.append({
        "Learning Rate": lr,
        "Accuracy": acc
    })


learning_rate_results_df = pd.DataFrame(
    learning_rate_results
)

print("\nEXPERIMENT 3: LEARNING RATE")
print(learning_rate_results_df)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\nACTIVATION FUNCTION RESULTS")
print(activation_results_df)

print("\nHIDDEN LAYER RESULTS")
print(hidden_layer_results_df)

print("\nLEARNING RATE RESULTS")
print(learning_rate_results_df)