"""
visualize_data.py
---------------------------------
Handles all visualization tasks for the College Admission Prediction dataset.
"""

import matplotlib.pyplot as plt


def plot_feature_distributions(df):
    """Plot histograms for numerical features."""
    features = ["GRE", "TOEFL", "University Rating", "SOP", "LOR", "CGPA", "Research"]
    titles = [
        "Distribution of GRE Scores",
        "Distribution of TOEFL Scores",
        "Distribution of University Rating",
        "Distribution of SOP",
        "Distribution of LOR",
        "Distribution of CGPA",
        "Distribution of Research Papers",
    ]

    for feat, title in zip(features, titles):
        if feat in df.columns:
            plt.figure(figsize=(6, 4))
            plt.hist(df[feat], rwidth=0.7, color="skyblue", edgecolor="black")
            plt.title(title)
            plt.xlabel(feat)
            plt.ylabel("Count")
            plt.grid(axis="y", alpha=0.5)
            plt.tight_layout()
            plt.show()
