"""
Iris Species Classification using Decision Tree Classifier
Dataset: UCI Iris Dataset (https://www.kaggle.com/datasets/uciml/iris)

This script demonstrates:
1. Data preprocessing (handling missing values, encoding labels)
2. Training a decision tree classifier
3. Model evaluation using accuracy, precision, and recall
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# STEP 1: LOAD THE DATASET

print("=" * 70)
print("IRIS SPECIES CLASSIFICATION - DECISION TREE CLASSIFIER")
print("=" * 70)
print("\n[STEP 1] Loading the Iris dataset...")

# Load the iris dataset from sklearn's built-in datasets
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Create a mapping for better readability
species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species_name'] = df['species'].map(species_names)

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())


# STEP 2: EXPLORATORY DATA ANALYSIS

print("\n" + "=" * 70)
print("[STEP 2] Exploratory Data Analysis")
print("=" * 70)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nClass Distribution:")
print(df['species_name'].value_counts())

print("\nChecking for missing values:")
missing_values = df.isnull().sum()
print(missing_values)
if missing_values.sum() == 0:
    print("✓ No missing values found!")
else:
    print("⚠ Missing values detected!")


# STEP 3: DATA PREPROCESSING

print("\n" + "=" * 70)
print("[STEP 3] Data Preprocessing")
print("=" * 70)

# Handle missing values (if any)
# Strategy: Fill numerical columns with median, categorical with mode
print("\nHandling missing values...")
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if df[column].dtype in ['float64', 'int64']:
            # Fill numerical columns with median
            df[column].fillna(df[column].median(), inplace=True)
            print(f"  - Filled {column} with median value")
        else:
            # Fill categorical columns with mode
            df[column].fillna(df[column].mode()[0], inplace=True)
            print(f"  - Filled {column} with mode value")

# Separate features and target
X = df[iris.feature_names]  # Features: sepal length, sepal width, petal length, petal width
y = df['species']  # Target: species (0, 1, 2)

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Label encoding (already done as iris.target is numeric, but showing the process)
print("\nLabel Encoding:")
print(f"  - Target variable is already encoded as:")
for idx, name in species_names.items():
    print(f"    {idx} = {name}")


# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS

print("\n" + "=" * 70)
print("[STEP 4] Splitting Data")
print("=" * 70)

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Training set distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"Testing set distribution:\n{pd.Series(y_test).value_counts().sort_index()}")


# STEP 5: TRAIN THE DECISION TREE CLASSIFIER

print("\n" + "=" * 70)
print("[STEP 5] Training Decision Tree Classifier")
print("=" * 70)

# Initialize the Decision Tree Classifier
# Parameters:
#   - criterion: The function to measure the quality of a split ('gini' or 'entropy')
#   - max_depth: Maximum depth of the tree (None means unlimited)
#   - min_samples_split: Minimum samples required to split an internal node
#   - random_state: For reproducibility
dt_classifier = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=2,
    random_state=42
)

print("Training the model...")
dt_classifier.fit(X_train, y_train)
print("✓ Model training completed!")

# Display feature importances
print("\nFeature Importances:")
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)


# STEP 6: MAKE PREDICTIONS

print("\n" + "=" * 70)
print("[STEP 6] Making Predictions")
print("=" * 70)

# Predict on training set
y_train_pred = dt_classifier.predict(X_train)
print(f"Training predictions generated: {len(y_train_pred)} samples")

# Predict on testing set
y_test_pred = dt_classifier.predict(X_test)
print(f"Testing predictions generated: {len(y_test_pred)} samples")


# STEP 7: MODEL EVALUATION

print("\n" + "=" * 70)
print("[STEP 7] Model Evaluation")
print("=" * 70)

# Evaluate on training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\n📊 TRAINING SET PERFORMANCE:")
print(f"  Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# Evaluate on testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\n📊 TESTING SET PERFORMANCE:")
print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Precision: Of all the instances predicted as a certain class, how many were correct?
# average='weighted' accounts for class imbalance
test_precision = precision_score(y_test, y_test_pred, average='weighted')
print(f"  Precision (weighted): {test_precision:.4f}")

# Recall: Of all the instances that actually belong to a class, how many did we identify?
# average='weighted' accounts for class imbalance
test_recall = recall_score(y_test, y_test_pred, average='weighted')
print(f"  Recall (weighted): {test_recall:.4f}")

# Per-class metrics
print(f"\n📊 PER-CLASS METRICS:")
precision_per_class = precision_score(y_test, y_test_pred, average=None)
recall_per_class = recall_score(y_test, y_test_pred, average=None)

for idx, name in species_names.items():
    print(f"\n  {name.upper()}:")
    print(f"    Precision: {precision_per_class[idx]:.4f}")
    print(f"    Recall: {recall_per_class[idx]:.4f}")

# Detailed classification report
print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_test_pred, target_names=list(species_names.values())))

# Confusion Matrix
print(f"\n📊 CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_test_pred)
print(f"{'':12} {'setosa':>10} {'versicolor':>12} {'virginica':>12}")
for idx, name in species_names.items():
    print(f"{name:12} {cm[idx][0]:10} {cm[idx][1]:12} {cm[idx][2]:12}")


# STEP 8: MODEL INSIGHTS

print("\n" + "=" * 70)
print("[STEP 8] Model Insights")
print("=" * 70)

print("\n🌳 Decision Tree Properties:")
print(f"  Tree depth: {dt_classifier.get_depth()}")
print(f"  Number of leaves: {dt_classifier.get_n_leaves()}")
print(f"  Number of features: {dt_classifier.n_features_in_}")

print("\n📝 Key Findings:")
print(f"  - Most important feature: {feature_importance.iloc[0]['feature']}")
print(f"  - Model achieved {test_accuracy*100:.2f}% accuracy on test set")
print(f"  - Training accuracy: {train_accuracy*100:.2f}%")

if train_accuracy > test_accuracy + 0.05:
    print(f"  ⚠ Potential overfitting detected (training accuracy significantly higher)")
else:
    print(f"  ✓ Model generalizes well to unseen data")


# STEP 9: SAMPLE PREDICTIONS

print("\n" + "=" * 70)
print("[STEP 9] Sample Predictions")
print("=" * 70)

print("\nMaking predictions on sample data:")
sample_indices = [0, 50, 100]
for idx in sample_indices:
    sample = X_test.iloc[idx:idx+1]
    actual = y_test.iloc[idx]
    prediction = dt_classifier.predict(sample)[0]
    probability = dt_classifier.predict_proba(sample)[0]

    print(f"\nSample {idx + 1}:")
    print(f"  Features: {sample.values[0]}")
    print(f"  Actual species: {species_names[actual]}")
    print(f"  Predicted species: {species_names[prediction]}")
    print(f"  Confidence: {probability[prediction]*100:.2f}%")
    print(f"  Probabilities: {', '.join([f'{species_names[i]}: {prob*100:.1f}%' for i, prob in enumerate(probability)])}")
    print(f"  Result: {'✓ CORRECT' if actual == prediction else '✗ INCORRECT'}")

# SUMMARY

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
✓ Dataset loaded and preprocessed successfully
✓ Decision Tree Classifier trained with {X_train.shape[0]} samples
✓ Model evaluated on {X_test.shape[0]} test samples

Final Metrics:
  - Accuracy: {test_accuracy*100:.2f}%
  - Precision: {test_precision*100:.2f}%
  - Recall: {test_recall*100:.2f}%

The decision tree successfully classifies iris species based on their
physical characteristics (sepal and petal measurements).
""")
print("=" * 70)
