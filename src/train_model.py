"""
train_model.py
---------------------------------
Trains and evaluates models for the College Admission Prediction project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


# ---------------------------
# 1. LOAD & PREPROCESS DATA
# ---------------------------
def load_data(path="data/admission_predict.csv"):
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    print(f" Data loaded successfully! Shape: {df.shape}")
    return df


def preprocess_data(df):
    """Clean and prepare data."""
    # Rename columns
    df = df.rename(columns={
        "GRE Score": "GRE",
        "TOEFL Score": "TOEFL",
        "LOR ": "LOR",
        "Chance of Admit ": "Probability"
    })

    # Drop Serial No. if exists
    if "Serial No." in df.columns:
        df.drop("Serial No.", axis=1, inplace=True)

    # Replace 0 with NaN for key numeric columns
    cols = ["GRE", "TOEFL", "University Rating", "SOP", "LOR", "CGPA"]
    df[cols] = df[cols].replace(0, np.nan)

    print(" Data cleaned successfully.")
    return df


# ---------------------------
# 2. MODEL COMPARISON
# ---------------------------
def compare_models(X, y):
    """Compare ML models using GridSearchCV."""
    models = {
        'linear_regression': {
            'model': LinearRegression(),
            'parameters': {}
        },
        'lasso': {
            'model': Lasso(),
            'parameters': {'alpha': [1, 2], 'selection': ['random', 'cyclic']}
        },
        'svr': {
            'model': SVR(),
            'parameters': {'gamma': ['auto', 'scale']}
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'parameters': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'parameters': {'n_estimators': [5, 10, 15, 20]}
        },
        'knn': {
            'model': KNeighborsRegressor(),
            'parameters': {'n_neighbors': [2, 5, 10, 20]}
        }
    }

    scores = []
    print(" Running GridSearchCV for each model...")
    for name, config in models.items():
        gs = GridSearchCV(config['model'], config['parameters'], cv=5, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': name,
            'best_parameters': gs.best_params_,
            'score': round(gs.best_score_, 4)
        })

    result = pd.DataFrame(scores)
    print("\n Model Comparison Results:\n", result)
    os.makedirs("results", exist_ok=True)
    result.to_csv("results/model_comparison.csv", index=False)
    print(" Saved model comparison results to results/model_comparison.csv")
    return result


# ---------------------------
# 3.  (LINEAR REG)
# ---------------------------
def train_linear_regression(X, y):
    """Train final Linear Regression model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    print(f" Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    r2 = r2_score(y_test, y_pred)
    print(f" Linear Regression R² Score: {round(r2, 3)}")

    # Cross-validation
    scores = cross_val_score(LinearRegression(), X, y, cv=5)
    print(f" Cross-validation Mean Score: {round(sum(scores)/len(scores), 3)}")

    # Plot actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", color="red")
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/actual_vs_predicted_lr.png", bbox_inches="tight")
    plt.close()
    print(" Saved Linear Regression results plot to results/actual_vs_predicted_lr.png")

    # Example Predictions
    ex1 = [[337, 118, 4, 4.5, 4.5, 9.65, 0]]
    ex2 = [[320, 113, 2, 2.0, 2.5, 8.64, 1]]
    p1 = round(model.predict(ex1)[0] * 100, 2)
    p2 = round(model.predict(ex2)[0] * 100, 2)
    print(f"\nExample Predictions:\n UCLA: {p1}%\n Alt Univ: {p2}%")

    # Save predictions
    with open("results/sample_predictions.txt", "w") as f:
        f.write(f"Prediction 1 (UCLA): {p1}%\n")
        f.write(f"Prediction 2: {p2}%\n")

    print(" Predictions saved to results/sample_predictions.txt")
    return model


# ---------------------------
# 4. MAIN PIPELINE
# ---------------------------
def main():
    df = load_data()
    df = preprocess_data(df)

    X = df.drop("Probability", axis=1)
    y = df["Probability"]

    compare_models(X, y)
    train_linear_regression(X, y)


if __name__ == "__main__":
    main()

