import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score,
    recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Load the dataset
file_path = 'heart_disease_dataset.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please ensure the dataset file exists in the current directory.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# ====================
# Data Overview
# ====================
print("Initial Data Overview:")
print(df.head())
print(df.info())
print(df.describe())
print("Dataset shape:", df.shape)

# Check for nulls and duplicates
print("Null values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Drop duplicates
df = df.drop_duplicates()

# ====================
# EDA (Exploratory Data Analysis)
# ====================
# Distribution of target variable
sns.countplot(data=df, x='target', hue='target', palette='viridis', legend=False)
plt.title('Target Variable Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot
selected_features = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
sns.pairplot(df[selected_features + ['target']], hue='target')
plt.show()

# Box plots for categorical features
categorical_features = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']
for feature in categorical_features:
    sns.boxplot(data=df, x=feature, y='age', hue='target')
    plt.title(f'{feature} vs Age')
    plt.show()

# Distribution of numeric features
numerical_features = selected_features
for feature in numerical_features:
    sns.histplot(data=df, x=feature, kde=True, hue='target')
    plt.title(f'Distribution of {feature}')
    plt.show()

# ====================
# Split data
# ====================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====================
# Model Training
# ====================
models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

results = []

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 2),
        'Precision': round(prec, 2),
        'Recall': round(rec, 2),
        'F1-Score': round(f1, 2)
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

# ====================
# Summary Table
# ====================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
print("\nModel Performance Summary:")
print(results_df)
