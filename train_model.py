"""Logistic regression on the Pima Indians Diabetes Dataset.

Predicts whether a patient has diabetes (Outcome = 1) from 8 medical measurements.
Run with:  python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load the data
df = pd.read_csv("diabetes.csv")
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# 2. Clean the data
# In these columns a value of 0 is medically impossible, so it really means "missing".
zero_means_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_means_missing] = df[zero_means_missing].replace(0, np.nan)
print("\nMissing values per column after cleaning:")
print(df[zero_means_missing].isna().sum())

# 3. Split features (X) and target (y)
X = df.drop(columns="Outcome")
y = df["Outcome"]

# 4. Train/test split (80/20). stratify=y keeps the diabetes ratio the same in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Build the model pipeline
# The pipeline fills missing values with the median, scales the features,
# then fits logistic regression. Doing this inside a pipeline means the medians
# are learned from the training data only (no data leakage from the test set).
model = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

# 6. Evaluate on the unseen test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_prob):.3f}")
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# 7. Which features matter most? (coefficients are comparable because we scaled)
coefs = pd.Series(model.named_steps["logreg"].coef_[0], index=X.columns)
print("Feature coefficients (largest first):")
print(coefs.sort_values(ascending=False).round(3))
