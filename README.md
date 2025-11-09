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



📊 Dataset Information

File: admission_predict.csv

Columns:

Column	Description
GRE Score	Graduate Record Examination (0–340)
TOEFL Score	English proficiency score (0–120)
University Rating	Reputation of the university (1–5)
SOP	Statement of Purpose strength (1–5)
LOR	Letter of Recommendation strength (1–5)
CGPA	Cumulative GPA (0–10)
Research	1 = Yes, 0 = No
Chance of Admit	Target variable (0–1)

🧹 Pre-processing Steps

Renamed columns for consistency (GRE Score → GRE, Chance of Admit → Probability)

Dropped Serial No.

Replaced 0 values with NaN in key numeric columns

🔧 Installation

If running locally:

git clone https://github.com/monarchbaweja1/College_Admission_Prediction.git
cd College_Admission_Prediction
pip install -r requirements.txt

🧾 requirements.txt (Colab-compatible versions)
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
scikit-learn==1.5.1
seaborn==0.13.2


(No need to install these on Colab; they’re pre-installed.)

▶️ How to Run on Google Colab

Open the notebook link above or upload College_Admission_Prediction.ipynb to Colab.

Mount your Google Drive to load the dataset:

from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/admission_predict.csv')


Run all cells sequentially:

Data Loading & Cleaning

Visualization

Model Training & Evaluation

Prediction

🧩 Project Workflow
Step	Description
1. Exploratory Data Analysis	View data shape, info, statistics, nulls
2. Visualization	Histograms for GRE, TOEFL, CGPA, etc.
3. Data Cleaning	Drop unused columns, handle missing values
4. Baseline Model	Train Random Forest Regressor for initial accuracy
5. Model Comparison	GridSearchCV over Linear Regression, Lasso, SVR, Decision Tree, Random Forest, KNN
6. Final Model	Train Linear Regression (highest cross-val score)
7. Prediction & Evaluation	Visualize Actual vs Predicted and predict new inputs
📈 Results Summary
Model	Best Parameters	Accuracy (R² Score)
Linear Regression	—	0.81
Random Forest	n_estimators = 100	0.78
Decision Tree	criterion = squared_error	0.73
KNN	n_neighbors = 5	0.69
Lasso Regression	alpha = 1	0.67
SVR	gamma = scale	0.64

✅ Final Model: Linear Regression (best generalization accuracy)

🧮 Example Predictions
# Input format: GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research
model.predict([[337, 118, 4, 4.5, 4.5, 9.65, 0]])
# → Predicted Admission Chance ≈ 92.7 %

model.predict([[320, 113, 2, 2.0, 2.5, 8.64, 1]])
# → Predicted Admission Chance ≈ 65.0 %

📊 Visualizations Included

GRE Distribution

TOEFL Distribution

University Rating, SOP, LOR, CGPA Histograms

Research Count Distribution

Actual vs Predicted Scatter Plots

Each chart provides quick insight into data spread and model performance.

🚀 Future Enhancements

Add feature importance & correlation heatmaps

Try XGBoost / Gradient Boosting models

Deploy via Streamlit or Flask Web App

Implement Hyperparameter Optimization (Optuna)

Integrate live user input for web prediction form
