# model/main.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
df = pd.read_csv('data/data.csv')

# Clean unused empty column (common in breast cancer dataset)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Encode diagnosis column
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Drop ID column if exists
df.drop(columns=['id'], errors='ignore', inplace=True)

# Separate features and labels
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Logistic Regression
logreg_grid = GridSearchCV(
    LogisticRegression(),
    param_grid={
        'C': [0.1, 1.0, 10],
        'solver': ['lbfgs', 'liblinear']
    },
    cv=5, scoring='accuracy', n_jobs=-1
)
logreg_grid.fit(X_train, y_train)
logreg_model = logreg_grid.best_estimator_

# Hyperparameter tuning for Random Forest
rf_grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid={
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    cv=5, scoring='accuracy', n_jobs=-1
)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_

# Hyperparameter tuning for SVM
svm_grid = GridSearchCV(
    SVC(probability=True),
    param_grid={
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    cv=5, scoring='accuracy', n_jobs=-1
)
svm_grid.fit(X_train, y_train)
svm_model = svm_grid.best_estimator_

# Save models in dictionary
models = {
    'logreg': logreg_model,
    'rf': rf_model,
    'svm': svm_model
}

# Store best parameters
best_params = {
    'logreg': logreg_grid.best_params_,
    'rf': rf_grid.best_params_,
    'svm': svm_grid.best_params_
}

# Dictionary to store metrics
metrics = {}
scores = {}

# Train and save each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for saving

    # Store individual metrics
    scores[name] = acc
    metrics[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    # Save model
    with open(f'model/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save scaler and imputer
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

# Save scores and metrics
with open('model/scores.pkl', 'wb') as f:
    pickle.dump(scores, f)
with open('model/metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

# Save best hyperparameters
with open('model/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print("✅ All models, scaler, imputer, scores, metrics, and best params saved successfully.")
