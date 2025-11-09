College-Prediction-Model <br>

Folder Structure <br>
College_Admission_Prediction/ <br>
│ <br>
├── data/ <br>
│ └── admission_predict.csv <br>
│ <br>
├── notebooks/ <br>
│ └── College_Admission_Prediction.ipynb <br>
│ <br>
├── src/ <br>
│ ├── train_model.py <br>
│ ├── visualize_data.py <br>
│ └── init.py <br>
│<br>
├── results/ <br>
│ ├── regression_actual_vs_predicted.png <br>
│ ├── calibration_curve.png <br>
│ ├── pdp_GRE.png <br>
│ ├── pdp_CGPA.png <br>
│ ├── linear_coefficients.csv <br>
│ └── model_comparison.csv <br>
│ <br>
├── requirements.txt <br>
├── README.md <br>
└── .gitignore <br>

** Dataset Information ** <br>

File: admission_predict.csv <br>
<br>
Columns: <br>
<br>
Column Description: <br>
GRE Score — Graduate Record Examination (0–340) <br>
TOEFL Score — English proficiency score (0–120) <br>
University Rating — Reputation of the university (1–5) <br>
SOP — Statement of Purpose strength (1–5) <br>
LOR — Letter of Recommendation strength (1–5) <br>
CGPA — Cumulative GPA (0–10) <br>
Research — 1 = Yes, 0 = No <br>
Chance of Admit — Target variable (0–1) <br>
<br>

** How to Run on Google Colab ** <br>

Open the notebook link above or upload College_Admission_Prediction.ipynb to Colab. <br>
<br>
Mount your Google Drive to load the dataset: <br>
<br>
from google.colab import drive <br>
drive.mount('/content/drive') <br>
df = pd.read_csv('/content/drive/MyDrive/admission_predict.csv') <br>
<br>
Run all cells sequentially: <br>
Data Loading & Cleaning <br>
Visualization <br>
Model Training & Evaluation <br>
Prediction & Fairness Check <br>

** Project Workflow ** <br>

Exploratory Data Analysis — View shape, info, statistics, and nulls <br>

Visualization — Histograms for GRE, TOEFL, CGPA, etc. <br>

Data Cleaning — Drop unused columns, handle missing values <br>

Regression Modeling — Train models (Linear, Lasso, Random Forest) using GridSearchCV <br>

Calibration — Apply CalibratedClassifierCV to improve probability estimates <br>

Threshold Policy — Define cost-sensitive thresholds for admission decisions <br>

Fairness Metrics — Evaluate subgroups (Research/University Rating) for bias <br>

Interpretability — Coefficients, Partial Dependence Plots (PDPs) <br>

Evaluation — Report RMSE, MAE, R², ROC-AUC, Brier Score <br>

Prediction — Compute real-world admission probability <br>

** 📈 Results Summary ** <br>

| Model | Best Parameters | Cross-Val R² | <br>
|--------|-----------------|--------------| <br>
| Linear Regression | — | 0.805 | <br>
| Lasso Regression | α = 0.1 | 0.79 | <br>
| Random Forest | n_estimators = 200, max_depth = 10 | 0.78 | <br>
<br>

Regression Metrics (on test set): <br>
RMSE ≈ 0.0562 <br>
MAE ≈ 0.0399 <br>
R² ≈ 0.8520 <br>
<br>

Calibration & Classification Metrics: <br>
ROC-AUC ≈ 0.9654 <br>
Brier Score ≈ 0.0806 <br>

<br>

Subgroup Fairness Metrics (Accuracy): <br>
Research=0 → 0.886 <br>
Research=1 → 0.857 <br>
UnivRating=5 → 0.957 <br>
UnivRating=3 → 0.839 <br>
<br>

** Final Model: Linear Regression + Calibrated Random Forest (best generalization & probability calibration)** <br>
** Example Predictions ** <br>

Input format: GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research <br>
model.predict([[337, 118, 4, 4.5, 4.5, 9.65, 0]]) <br>
→ Predicted Admission Chance ≈ 93.2 % <br>
<br>
model.predict([[320, 113, 2, 2.0, 2.5, 8.64, 1]]) <br>
→ Predicted Admission Chance ≈ 72.4 % <br>

** Visualizations Included ** <br>

GRE Distribution <br>
TOEFL Distribution <br>
University Rating, SOP, LOR, CGPA Histograms <br>
Research Count Distribution <br>
Regression Actual vs Predicted Scatter <br>
Calibration Curve (Isotonic Regression) <br>
Partial Dependence Plots for GRE & CGPA <br>
Linear Regression Coefficients (Feature Importance) <br>

Each visualization provides insight into data patterns, prediction quality, and fairness across subgroups. <br>

** Future Enhancements ** <br>

Add SHAP-based feature importance for deeper interpretability <br>
Implement advanced boosting (XGBoost, LightGBM) models <br>
Deploy via Streamlit / Flask for live prediction <br>
Add Optuna or Bayesian Optimization for hyperparameter tuning <br>
Automate fairness dashboards using Plotly / Dash <br>
Integrate continuous retraining pipeline with version tracking <br>

** Author ** <br>

Monarch Baweja <br>
Goa Institute of Management <br>
GitHub: monarchbaweja1
 <br>

** License ** <br>

MIT License — Open for educational and research use. <br>
