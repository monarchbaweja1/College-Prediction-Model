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
