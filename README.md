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

