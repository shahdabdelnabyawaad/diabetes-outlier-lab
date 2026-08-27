import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Diabetes Outlier Lab", layout="wide")

# ============================================
# Load Data
# ============================================
df = pd.read_csv("data/diabetes.csv")

zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
numeric_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "Age"]

# ============================================
# Sidebar
# ============================================
st.sidebar.header("Control Panel")
selected_col = st.sidebar.selectbox("Select feature:", numeric_cols)
detection_method = st.sidebar.selectbox("Detection method:", ["IQR Method", "Z-Score Method"])
treatment = st.sidebar.selectbox(
    "Treatment strategy:",
    ["None (Keep)", "Trim (Remove)", "Winsorize (Cap)", "Impute (Median)", "Transform (Log)"]
)

# ============================================
# Header
# ============================================
st.title(" 🩺 Diabetes Outlier Detection Studio")
st.caption("An applied statistics tool for detecting and treating anomalies in clinical data.")

tab1, tab2, tab3 = st.tabs(["Feature Explorer", "Full Dataset Overview", "Research Notes"])

# ============================================
# Shared calculations
# ============================================
Q1 = df[selected_col].quantile(0.25)
Q3 = df[selected_col].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
extreme_lower = Q1 - 3 * IQR
extreme_upper = Q3 + 3 * IQR

mean_val = df[selected_col].mean()
std_val = df[selected_col].std()
zscore = (df[selected_col] - mean_val) / std_val

if detection_method == "IQR Method":
    outlier_mask = (df[selected_col] < lower_fence) | (df[selected_col] > upper_fence)
else:
    outlier_mask = (zscore > 3) | (zscore < -3)

outliers = df[outlier_mask]
outlier_pct = round(outliers.shape[0] / len(df) * 100, 2)

# ============================================
# TAB 1: Feature Explorer
# ============================================
with tab1:
    with st.expander("Preview raw data (first 10 rows)"):
        st.dataframe(df.head(10))

    st.subheader("Analyzing: " + selected_col)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Outliers Detected", outliers.shape[0])
    c3.metric("Outlier Percentage", str(outlier_pct) + "%")

    if selected_col in zero_as_missing:
        zero_count = (df[selected_col] == 0).sum()
        st.info(
            "Note: " + str(zero_count) + " rows have a value of 0 in this column, "
            "which may indicate missing data rather than a true measurement."
        )

    st.divider()
    st.subheader("Distribution (" + detection_method + ")")

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df[selected_col],
        name=selected_col,
        boxpoints="outliers",
        marker_color="#0F766E",
    ))
    fig.update_layout(
        template="plotly_white",
        height=450,
        yaxis_title=selected_col,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Q1:", round(Q1, 2), " | Q3:", round(Q3, 2), " | IQR:", round(IQR, 2))
    st.write("Lower Fence:", round(lower_fence, 2), " | Upper Fence:", round(upper_fence, 2))
    st.write("Extreme Lower Fence:", round(extreme_lower, 2), " | Extreme Upper Fence:", round(extreme_upper, 2))

# ============================================
# TAB 2: Full Dataset Overview
# ============================================
with tab2:
    st.subheader("Outlier Summary Across All Features")

    summary_rows = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lf = q1 - 1.5 * iqr
        uf = q3 + 1.5 * iqr
        count = df[(df[col] < lf) | (df[col] > uf)].shape[0]
        summary_rows.append({
            "Feature": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Outliers (IQR)": count,
            "% of Data": round(count / len(df) * 100, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Apply Treatment to: " + selected_col)

    df_treated = df.copy()

    if treatment == "None (Keep)":
        st.write("No changes applied.")

    elif treatment == "Trim (Remove)":
        df_treated = df_treated[~outlier_mask]
        st.write("Rows removed:", df.shape[0] - df_treated.shape[0])

    elif treatment == "Winsorize (Cap)":
        df_treated[selected_col] = df_treated[selected_col].clip(lower=lower_fence, upper=upper_fence)
        st.write("Values capped between", round(lower_fence, 2), "and", round(upper_fence, 2))

    elif treatment == "Impute (Median)":
        median_val = df_treated[selected_col].median()
        df_treated.loc[outlier_mask, selected_col] = median_val
        st.write("Outliers replaced with median:", round(median_val, 2))

    elif treatment == "Transform (Log)":
        df_treated[selected_col] = df_treated[selected_col].apply(lambda x: 0 if x <= 0 else x)
        df_treated[selected_col] = np.log1p(df_treated[selected_col])
        st.write("Applied log(1 + x) transformation.")

    fig2 = go.Figure()
    fig2.add_trace(go.Box(y=df[selected_col], name="Before", marker_color="#94A3B8"))
    fig2.add_trace(go.Box(y=df_treated[selected_col], name="After", marker_color="#0F766E"))
    fig2.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig2, use_container_width=True)

# ============================================
# TAB 3: Research Notes
# ============================================
with tab3:
    st.subheader("Theoretical Background")

    st.markdown("**What is an outlier?**")
    st.write(
        "An observation that lies at an abnormal distance from the rest of the data. "
        "It may indicate a data entry error, a measurement fault, or a genuine rare event."
    )

    st.markdown("**IQR Method**")
    st.latex(r"IQR = Q_3 - Q_1")
    st.latex(r"\text{Lower Fence} = Q_1 - 1.5 \times IQR")
    st.latex(r"\text{Upper Fence} = Q_3 + 1.5 \times IQR")
    st.latex(r"\text{Extreme Fences} = Q_1 - 3 \times IQR \ \text{,} \ Q_3 + 3 \times IQR")

    st.markdown("**Z-Score Method**")
    st.latex(r"Z = \frac{x - \mu}{\sigma}")
    st.write("A value is typically flagged as an outlier when |Z| > 3.")

    st.markdown("**Treatment Strategies**")
    st.write("- **Keep:** the value is a genuine, valid extreme case.")
    st.write("- **Trim:** remove the row when it is clearly an error.")
    st.write("- **Winsorize:** cap the value at the fence, keeping the row.")
    st.write("- **Impute:** replace with the median or another estimate.")
    st.write("- **Transform:** apply a log transform to reduce skew.")

    st.markdown("**References**")
    st.write("Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.")
    st.write("Barnett, V., & Lewis, T. (1994). Outliers in Statistical Data (3rd ed.). Wiley.")
    st.write("Hodge, V. J., & Austin, J. (2004). A Survey of Outlier Detection Methodologies. Artificial Intelligence Review, 22(2), 85-126.")