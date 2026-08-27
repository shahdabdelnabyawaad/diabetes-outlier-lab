# 🩺 Diabetes Outlier Detection & Treatment Lab

An interactive Streamlit lab for detecting and treating outliers in the
diabetes screening dataset, using both the IQR (Tukey fence) method and the
Z-score method.

## Features
- **Single Feature Deep-Dive** — pick one numeric column, see Q1/Q3/IQR,
  fences, flagged outliers, and a before/after boxplot for any treatment.
- **Full Dataset Batch Studio** — scan every numeric column at once and run
  a global treatment across the whole dataset.
- **Theoretical Research Guide** — the statistics behind the app: IQR vs
  Z-score, treatment strategies, and a diabetes-specific note on
  zero-as-missing values (`Glucose`, `BloodPressure`, `SkinThickness`,
  `Insulin`, `BMI` cannot really be 0 in a living patient).

## Project structure
```
diabetes-outlier-lab/
├── app.py                  # Streamlit app (UI + tabs)
├── stats_engine.py         # StatisticalDetector class (IQR, Z-score, treatments)
├── data/
│   └── diabetes.csv        # placeholder dataset — replace with your real data
├── requirements.txt
├── .streamlit/config.toml  # light teal theme
└── .gitignore
```

## Dataset
`data/diabetes.csv` is the Pima Indians Diabetes dataset (768 rows, 9
columns: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`,
`Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, `Outcome`), sourced from
[plotly/datasets](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv).
To use a different dataset, drop your own CSV in `data/` (or use the
in-app "Upload Custom Dataset" uploader) and update `ZERO_AS_MISSING_COLS`
in `app.py` if your column names differ.

## Run it locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Push this to GitHub
1. Create an empty repo on GitHub (no README/gitignore, so it stays empty) —
   e.g. `diabetes-outlier-lab`.
2. From inside this project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: diabetes outlier detection lab"
   git branch -M main
   git remote add origin https://github.com/<your-username>/diabetes-outlier-lab.git
   git push -u origin main
   ```
3. If you use SSH instead of HTTPS, use
   `git@github.com:<your-username>/diabetes-outlier-lab.git` for the remote.

## Deploy (optional)
Once pushed, you can deploy it for free on
[Streamlit Community Cloud](https://streamlit.io/cloud) by connecting your
GitHub repo and pointing it at `app.py`.
