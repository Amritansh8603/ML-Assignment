# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Optional: Uncomment these if available
# from imblearn.over_sampling import SMOTE
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
df = pd.read_csv("Fraud.csv")

# Drop unnecessary columns
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Encode 'type' using one-hot encoding
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Check for and remove outliers (optional, shown for 'amount')
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['amount'] >= Q1 - 1.5 * IQR) & (df['amount'] <= Q3 + 1.5 * IQR)]

# Define X and y
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Handle imbalance with SMOTE
# smote = SMOTE(random_state=42)
# X_scaled, y = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Feature importances
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 5 Important Features:\n", importances.sort_values(ascending=False).head())
