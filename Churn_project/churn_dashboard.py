# ============================================================
# Customer Churn Prediction — Streamlit Dashboard
# Upload ANY CSV → Auto-detect → Train Models → Dashboard
# Batch A15 | CSE-A | Guide: Dr. P. Vamsi Krishna
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# ─── Page Config ───
st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📊", layout="wide")

st.title("Customer Churn Predictive Analysis")


# ============================================================
# STEP 1 — UPLOAD CSV
# ============================================================
uploaded_file = st.file_uploader("Upload any Customer Churn CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to get started. The system will auto-detect columns and run the full ML pipeline.")
    st.markdown("""
    **Supported datasets:** Netflix, Telco, Banking, SaaS, E-commerce — any CSV with a churn column.

    **What happens after upload:**
    1. Auto-detects target (churn) column, ID column, categorical & numeric columns
    2. Cleans data & handles missing values
    3. Applies SMOTE for class balancing
    4. Trains Logistic Regression + Random Forest
    5. Shows full interactive dashboard with results
    """)
    st.stop()

# ============================================================
# STEP 2 — LOAD & AUTO-DETECT
# ============================================================

@st.cache_data(show_spinner="Loading dataset...")
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

df_raw = load_csv(uploaded_file.getvalue())
df = df_raw.copy()

st.success(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# --- Auto-detect Target Column ---
churn_col_candidates = [c for c in df.columns if 'churn' in c.lower()]
if churn_col_candidates:
    TARGET_COL = churn_col_candidates[0]
else:
    # Let user pick if no churn column found
    TARGET_COL = st.selectbox("No 'churn' column found. Select the target column:", df.columns)

# --- Auto-detect ID Column ---
id_col_candidates = [c for c in df.columns if 'id' in c.lower() and df[c].nunique() == len(df)]
if id_col_candidates:
    ID_COL = id_col_candidates[0]
    customer_ids = df[ID_COL].copy()
    df.drop(ID_COL, axis=1, inplace=True)
else:
    ID_COL = None
    customer_ids = pd.Series(range(len(df)), name="CustomerIndex")

# ============================================================
# STEP 3 — PREPROCESSING (runs automatically)
# ============================================================

# --- Handle Missing Values ---
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# --- Encode Target ---
if df[TARGET_COL].dtype == 'object':
    unique_vals = df[TARGET_COL].unique()
    churn_labels = [v for v in unique_vals if any(kw in str(v).lower() for kw in ['yes', 'churn', 'true', '1', 'leave', 'cancel'])]
    no_churn_labels = [v for v in unique_vals if v not in churn_labels]
    if not churn_labels:
        churn_labels = [unique_vals[0]]
        no_churn_labels = [v for v in unique_vals if v not in churn_labels]
    mapping = {v: 1 for v in churn_labels}
    mapping.update({v: 0 for v in no_churn_labels})
    df[TARGET_COL] = df[TARGET_COL].map(mapping)

# --- Encode Categorical Features ---
le_dict = {}
categorical_cols = df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# --- Split Features & Target ---
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# --- Scale Numeric Features ---
numeric_cols = [col for col in X.columns if X[col].nunique() > 10]
scaler = StandardScaler()
if numeric_cols:
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ============================================================
# STEP 4 — TRAIN-TEST SPLIT + SMOTE + MODEL TRAINING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_bal, y_train_bal)
lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train_bal, y_train_bal)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 5 — COMPUTE METRICS
# ============================================================
def get_metrics(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_prob)
    }

lr_metrics = get_metrics("Logistic Regression", y_test, lr_pred, lr_proba)
rf_metrics = get_metrics("Random Forest", y_test, rf_pred, rf_proba)
comparison = pd.DataFrame([lr_metrics, rf_metrics])
best_model_name = comparison.set_index("Model")["F1 Score"].idxmax()

# Risk categories
risk_labels = pd.cut(rf_proba, bins=[0, 0.3, 0.6, 1.0],
                     labels=["Low Risk", "Medium Risk", "High Risk"], include_lowest=True)

# ============================================================
# =================== DASHBOARD STARTS ======================
# ============================================================
st.markdown("---")

# ─── SECTION 1: KPI CARDS ───
st.subheader("Key Metrics")
total = len(y_test)
actual_churned = int(y_test.sum())
predicted_churned = int(rf_pred.sum())
high_risk_count = int((risk_labels == "High Risk").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers (Test)", total)
c2.metric("Actual Churned", actual_churned)
c3.metric("Churn Rate", f"{actual_churned / total * 100:.1f}%")
c4.metric("High Risk", high_risk_count)

st.markdown("---")

# ─── SECTION 2: MODEL COMPARISON ───
st.subheader("Model Performance Comparison")

col_l, col_r = st.columns(2)

with col_l:
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    comp_melted = comparison.melt(id_vars="Model", value_vars=metric_names,
                                  var_name="Metric", value_name="Score")
    comp_melted["Score_pct"] = comp_melted["Score"] * 100
    fig_comp = px.bar(comp_melted, x="Metric", y="Score_pct", color="Model",
                      barmode="group", text_auto=".1f",
                      color_discrete_sequence=["#3498db", "#e74c3c"],
                      labels={"Score_pct": "Score (%)"})
    fig_comp.update_layout(title="LR vs RF Performance", yaxis_range=[0, 105], height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

with col_r:
    st.markdown("#### Detailed Metrics")
    disp = comparison.copy()
    for m in metric_names:
        disp[m] = disp[m].apply(lambda x: f"{x * 100:.2f}%")
    st.dataframe(disp, use_container_width=True, hide_index=True)
    best_f1 = f"{comparison.set_index('Model').loc[best_model_name, 'F1 Score'] * 100:.2f}%"
    st.success(f"Best Model: **{best_model_name}** (F1: {best_f1})")

st.markdown("---")

# ─── SECTION 3: CHURN & RISK DISTRIBUTION ───
st.subheader("Churn Analysis")

ca, cb, cc = st.columns(3)

with ca:
    pred_labels = pd.Series(rf_pred).map({1: "Churn", 0: "No Churn"})
    dist = pred_labels.value_counts()
    fig_pie = px.pie(values=dist.values, names=dist.index,
                     color_discrete_sequence=["#2ecc71", "#e74c3c"],
                     title="Predicted Churn Split")
    fig_pie.update_traces(textinfo="percent+label+value")
    st.plotly_chart(fig_pie, use_container_width=True)

with cb:
    risk_dist = risk_labels.value_counts()
    cmap = {"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
    fig_risk = px.pie(values=risk_dist.values, names=risk_dist.index,
                      color=risk_dist.index, color_discrete_map=cmap,
                      title="Risk Distribution", hole=0.4)
    fig_risk.update_traces(textinfo="percent+label+value")
    st.plotly_chart(fig_risk, use_container_width=True)

with cc:
    fig_hist = px.histogram(x=rf_proba, nbins=30, color_discrete_sequence=["#3498db"],
                            title="Churn Probability Distribution",
                            labels={"x": "Churn Probability", "y": "Count"})
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ─── SECTION 4: ROC CURVES ───
st.subheader("ROC Curves")

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, name=f"LR (AUC={auc(lr_fpr, lr_tpr):.3f})",
                             line=dict(color="#3498db", width=2)))
fig_roc.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, name=f"RF (AUC={auc(rf_fpr, rf_tpr):.3f})",
                             line=dict(color="#e74c3c", width=2)))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                             line=dict(color="gray", dash="dash")))
fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      title="ROC Curve Comparison", height=400)
st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("---")

# ─── SECTION 5: CONFUSION MATRICES ───
st.subheader("Confusion Matrices")

cm_l, cm_r = st.columns(2)

for col_plot, name, preds in [(cm_l, "Logistic Regression", lr_pred), (cm_r, "Random Forest", rf_pred)]:
    with col_plot:
        cm = confusion_matrix(y_test, preds)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           x=["No Churn", "Churn"], y=["No Churn", "Churn"],
                           title=f"{name} — Confusion Matrix",
                           labels={"x": "Predicted", "y": "Actual"})
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# ─── SECTION 6: FEATURE IMPORTANCE ───
st.subheader("Feature Importance")

fi_l, fi_r = st.columns(2)

with fi_l:
    imp = pd.DataFrame({"Feature": X.columns, "Importance": rf_model.feature_importances_})
    imp = imp.sort_values("Importance", ascending=True)
    fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="RdYlGn_r",
                     title="Random Forest — Feature Importance")
    fig_imp.update_layout(height=max(350, len(X.columns) * 30), showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

with fi_r:
    coeff = pd.DataFrame({"Feature": X.columns, "Coefficient": lr_model.coef_[0]})
    coeff = coeff.sort_values("Coefficient", ascending=True)
    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in coeff["Coefficient"]]
    fig_coeff = go.Figure(go.Bar(x=coeff["Coefficient"], y=coeff["Feature"],
                                  orientation="h", marker_color=colors))
    fig_coeff.update_layout(title="LR — Feature Coefficients",
                            xaxis_title="Coefficient (Red = increases churn)",
                            height=max(350, len(X.columns) * 30))
    st.plotly_chart(fig_coeff, use_container_width=True)

st.markdown("---")

# ─── SECTION 7: CHURN BY CATEGORY (from raw data) ───
st.subheader("Churn by Category")

raw_cat_cols = [c for c in df_raw.select_dtypes(include="object").columns
                if df_raw[c].nunique() <= 20
                and c != TARGET_COL
                and (ID_COL is None or c != ID_COL)]

if raw_cat_cols:
    selected = st.selectbox("Select feature to analyze:", raw_cat_cols)
    # Use test set indices
    test_raw = df_raw.loc[X_test.index].copy()
    test_raw["Prediction"] = pd.Series(rf_pred, index=X_test.index).map({1: "Churn", 0: "No Churn"})
    ct = pd.crosstab(test_raw[selected], test_raw["Prediction"])
    fig_cat = px.bar(ct, barmode="group", color_discrete_sequence=["#2ecc71", "#e74c3c"],
                     title=f"Churn by {selected}")
    fig_cat.update_layout(xaxis_title=selected, yaxis_title="Count", height=400)
    st.plotly_chart(fig_cat, use_container_width=True)

st.markdown("---")

# ─── SECTION 8: HIGH RISK CUSTOMERS TABLE ───
st.subheader("High Risk Customers")

results = df_raw.loc[X_test.index].copy()
results["RF_Prediction"] = pd.Series(rf_pred, index=X_test.index).map({1: "Churn", 0: "No Churn"})
results["Churn_Probability"] = pd.Series(rf_proba, index=X_test.index).round(4)
results["Risk_Category"] = np.array(risk_labels)

high_risk = results[results["Risk_Category"] == "High Risk"].sort_values("Churn_Probability", ascending=False)
st.write(f"Showing **{len(high_risk)}** high-risk customers")
st.dataframe(high_risk.head(50), use_container_width=True, hide_index=True)

st.markdown("---")

# ─── SECTION 9: DOWNLOAD RESULTS ───
st.subheader("Download Results")

dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("Download Predictions CSV",
                       results.to_csv(index=False), "churn_predictions.csv", "text/csv")
with dl2:
    st.download_button("Download Model Comparison CSV",
                       comparison.to_csv(index=False), "model_comparison.csv", "text/csv")
with dl3:
    imp_export = pd.DataFrame({"Feature": X.columns, "RF_Importance": rf_model.feature_importances_,
                                "LR_Coefficient": lr_model.coef_[0]})
    st.download_button("Download Feature Importance CSV",
                       imp_export.to_csv(index=False), "feature_importance.csv", "text/csv")

# ─── Footer ───
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "Customer Churn Predictive Analysis — Batch A15, CSE-A<br>"
    "Guide: Dr. P. Vamsi Krishna | Vignan's Institute of Engineering for Women"
    "</div>",
    unsafe_allow_html=True
)
