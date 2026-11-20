# Diabetes Prediction with Logistic Regression

A machine learning project that predicts whether a patient has diabetes using the Pima Indians Diabetes Dataset. It covers data cleaning, model training, and evaluation in a single Python script.

## Dataset

`diabetes.csv` has 768 patient records, 8 input features, and 1 target column.

| Column | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes family-history score |
| Age | Age in years |
| **Outcome** | **Target: 1 = diabetes, 0 = no diabetes** |

## Approach

1. **Data cleaning.** In `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`, a value of 0 is not physically possible. These zeros are really missing values, so I replaced them with `NaN`.
2. **Imputation.** Missing values are filled with the column median, which is robust to outliers.
3. **Scaling.** Features are standardized with `StandardScaler` so the model coefficients are comparable.
4. **Train/test split.** 80/20 split, stratified on `Outcome` so both sets keep the same share of diabetic patients.
5. **Model.** `LogisticRegression` from scikit-learn.
6. **No data leakage.** Imputation and scaling live inside a scikit-learn `Pipeline`, so they are fit on the training data only.

## Evaluation

The model is evaluated on the held-out test set using:

- Accuracy
- ROC AUC
- Confusion matrix
- Precision, recall, and F1 (classification report)
- Feature coefficients, to see which measurements drive the prediction

### Results

| Metric | Score |
|---|---|
| Accuracy | 0.708 |
| ROC AUC | 0.813 |

## How to Run

```bash
pip install pandas numpy scikit-learn
python train_model.py
```

Make sure `diabetes.csv` is in the same folder as the script.

## Possible Next Steps

- Cross-validation for a more reliable performance estimate
- Compare against Random Forest or Gradient Boosting
- Tune the decision threshold to favor recall, since missing a diabetic patient is costlier than a false alarm

## Tech Stack

Python, pandas, NumPy, scikit-learn
