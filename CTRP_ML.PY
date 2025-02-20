import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the dataset path
file_path = r'C:\Users\shams\Downloads\INNOVATE\dataset.csv'

# Check if the dataset file exists
if os.path.exists(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(data.head())
else:
    print(f"Error: Dataset not found at {file_path}. Please ensure the file is in the correct location.")
    exit()

# Dataset Information
print("\nDataset Information:")
print(data.info())

print("\nDataset Statistics:")
print(data.describe())

# Visualize target variable distribution
sns.countplot(x='Clicked on Ad', data=data)
plt.title('Distribution of Clicked vs Not Clicked')
plt.xlabel('Clicked on Ad (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.show()

# Handle missing values
data.fillna(0, inplace=True)

# Drop irrelevant columns if present
irrelevant_columns = ['Ad Topic Line', 'City', 'Country', 'Timestamp']  # Example of irrelevant columns
data.drop(irrelevant_columns, axis=1, inplace=True)

# Encode categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    data = pd.get_dummies(data, columns=categorical_columns)
    print("\nCategorical variables encoded.")

# Split data into features and target
X = data.drop('Clicked on Ad', axis=1)  # Use 'Clicked on Ad' as the target column
y = data['Clicked on Ad']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into training and testing sets.")

# Train the Random Forest model
print("\nTraining the Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Model evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")

# Feature importance visualization
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()