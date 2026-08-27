# 🩺 Diabetes Outlier Detection & Treatment Lab

# Diabetes Outlier Detection Studio

An interactive Streamlit application for detecting, visualizing, and treating
outliers in real-world clinical data, using the Pima Indians Diabetes dataset
as a case study.

## Overview

Outliers are observations that differ significantly from the rest of a dataset.
Some outliers are data entry or measurement errors; others represent rare but
genuine and clinically important cases. This project treats outlier detection
as an interactive, exploratory process rather than an automatic deletion step.

The app lets a user:

- Select any numeric feature from the dataset (Glucose, BMI, BloodPressure, etc.)
- Detect outliers using either the IQR method or the Z-Score method
- View the calculated Q1, Q3, IQR, and fence boundaries
- Apply a treatment strategy: Keep, Trim, Winsorize, Impute, or Log Transform
- Compare the distribution before and after treatment with an interactive boxplot
- Read a short research-notes section covering the statistical background

## Research Question

How can different outlier detection and treatment strategies influence the
statistical interpretation of real-world data?

This question is explored using the diabetes dataset, where several columns
(Glucose, BloodPressure, SkinThickness, Insulin, BMI) use 0 to represent a
missing reading — a physiologically impossible value that the IQR method
correctly flags as an outlier.

## Detection Methods

**IQR (Interquartile Range)**

IQR = Q3 - Q1
Lower Fence = Q1 - 1.5 * IQR
Upper Fence = Q3 + 1.5 * IQR
Extreme Lower Fence = Q1 - 3 * IQR
Extreme Upper Fence = Q3 + 3 * IQR


**Z-Score**
Z = (x - mean) / std
A value is flagged as an outlier when |Z| > 3.

## Treatment Strategies

| Strategy | Description |
|---|---|
| Keep | No changes; the value is treated as valid |
| Trim | Removes rows containing outliers |
| Winsorize | Caps outlier values at the fence boundary |
| Impute | Replaces outliers with the column median |
| Transform | Applies a log(1 + x) transformation to reduce skew |

## Project Structure
├── app.py # Main Streamlit application
├── data/
│ └── diabetes.csv # Pima Indians Diabetes dataset
├── .streamlit/
│ └── config.toml # App theme configuration
└── README.md

## Dataset

Pima Indians Diabetes Dataset — 768 patient records, 8 numeric clinical
features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age).

## References

- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Barnett, V., & Lewis, T. (1994). *Outliers in Statistical Data* (3rd ed.). Wiley.
- Hodge, V. J., & Austin, J. (2004). A Survey of Outlier Detection Methodologies.
  *Artificial Intelligence Review*, 22(2), 85–126.
  