# 🎓 Admissions Predictor (Streamlit)

This repo runs a Streamlit app that:
- predicts admission probability + decision (thresholded)
- batch scores uploaded CSVs
- optionally evaluates metrics if `admit_label` exists
- auto-trains a model if `artifacts/admissions_model.pkl` is missing

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
What happens on first run?
If no model exists, the app will:

generate a synthetic dataset (~21,534 rows)

train multiple models + pick best on validation

calibrate probabilities

save the model bundle to artifacts/admissions_model.pkl

Input CSV format (for batch scoring)
Your CSV can contain a subset of columns. Recommended columns:

term, program, major, country, uni_tier

cgpa_10, gre_total, gre_aw, toefl, ielts

work_exp_years, internships, projects, research_exp, publications

sop_score, lor_score, portfolio_score, interview_score

gender, income_bucket, first_gen

If admit_label exists, the Evaluate tab will compute metrics.

python
Copy code

---
