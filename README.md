# College-Prediction-Model

Folder Structure   <br>
College_Admission_Prediction/         <br>
│    <br>
├── data/    <br> 
│   └── admission_predict.csv    <br>     
│         <br>
├── notebooks/        <br>
│   └── College_Admission_Prediction.ipynb   <br>
│       <br>
├── src/  <br>
│   ├── train_model.py <br>
│   └── __init__.py   <br>
│<br>
├── results/            <br>
│   └── .gitkeep          <br>
│           <br>
├── requirements.txt         <br>
├── README.md        <br>
└── .gitignore           <br>



# ** Dataset Information  **            

File: admission_predict.csv      <br>
<br>
Columns:  <br>

**Column	Description:** <br>
GRE Score	Graduate Record Examination (0–340)   <br>
TOEFL Score	English proficiency score (0–120)   <br>
University Rating	Reputation of the university (1–5)   <br>
SOP	Statement of Purpose strength (1–5)  <br>
LOR	Letter of Recommendation strength (1–5)   <br>
CGPA	Cumulative GPA (0–10)   <br>
Research	1 = Yes, 0 = No   <br>
Chance of Admit	Target variable (0–1)    <br>
<br>

#** How to Run on Google Colab**

Open the notebook link above or upload College_Admission_Prediction.ipynb to Colab.   ****
<br>
Mount your Google Drive to load the dataset:  <br>
<br>
from google.colab import drive    <br>
drive.mount('/content/drive')   <br>
df = pd.read_csv('/content/drive/MyDrive/admission_predict.csv')   <br>
<br>
<br>
**Run all cells sequentially:**
<br>

Data Loading & Cleaning   <br>
Visualization  <br>
Model Training & Evaluation   ****
Prediction   <br>

# ** Project Workflow**
<br>
1. Exploratory Data Analysis	View data shape, info, statistics, nulls   <br>
2. Visualization	Histograms for GRE, TOEFL, CGPA, etc.   <br>
3. Data Cleaning	Drop unused columns, handle missing values   <br>
4. Baseline Model	Train Random Forest Regressor for initial accuracy <br>
5. Model Comparison	GridSearchCV over Linear Regression, Lasso, SVR, Decision Tree, Random Forest, KNN   <br>
6. Final Model	Train Linear Regression (highest cross-val score)   <br>
7. Prediction & Evaluation	Visualize Actual vs Predicted and predict new inputs   <br>
# **📈 Results Summary** 
Model	Best Parameters	Accuracy (R² Score)   <br>
Linear Regression	—	0.81   <br>
Random Forest	n_estimators = 100	0.78   <br>
Decision Tree	criterion = squared_error	0.73   <br>
KNN	n_neighbors = 5	0.69   <br>
Lasso Regression	alpha = 1	0.67    <br>
SVR	gamma = scale	0.64    <br>
<br>
# ** Final Model: Linear Regression (best generalization accuracy)**

# ** Example Predictions**<br>
# Input format: GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research  <br>
model.predict([[337, 118, 4, 4.5, 4.5, 9.65, 0]])   <br>
# → Predicted Admission Chance ≈ 92.855 %  <br>
<br>
model.predict([[320, 113, 2, 2.0, 2.5, 8.64, 1]])  <br>
# → Predicted Admission Chance ≈ 73.627 %   <br>
<br>
# ** Visualizations Included**
<br>
GRE Distribution  <br>
<br>
TOEFL Distribution   <br>

University Rating, SOP, LOR, CGPA Histograms   <br>

Research Count Distribution   <br>

Actual vs Predicted Scatter Plots   <br>

Each chart provides quick insight into data spread and model performance.   <br>

# ** Future Enhancements**

Add feature importance & correlation heatmaps   <br>

Try XGBoost / Gradient Boosting models   <br>

Deploy via Streamlit or Flask Web App   <br>

Implement Hyperparameter Optimization (Optuna)   <br>

Integrate live user input for web prediction form
