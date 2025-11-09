"""
visualize_data.py
---------------------------------
EDA & visualization utilities for College Admission Prediction dataset.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_distributions(df: pd.DataFrame):
    """Plot distributions for numeric features."""
    features = ["GRE", "TOEFL", "University Rating", "SOP", "LOR", "CGPA", "Research"]

    for col in features:
        if col not in df.columns:
            continue
        plt.hist(df[col].dropna(), bins=20, rwidth=0.8, edgecolor="black", color="skyblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.show()
