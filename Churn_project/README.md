# Customer Churn Predictive Analysis Dashboard

> **Batch A15 | CSE-A | Guide: Dr. P. Vamsi Krishna**
> Vignan's Institute of Engineering for Women

A Streamlit-based interactive dashboard that predicts customer churn using Machine Learning. Upload any customer CSV dataset and get instant predictions, visualizations, and downloadable reports.

## Reference Paper

**"Customer Churn Prediction: A Systematic Review of Recent Advances, Trends, and Challenges in Machine Learning and Deep Learning"**
— Imani et al. (2025), *Machine Learning and Knowledge Extraction*, MDPI
DOI: `https://doi.org/10.3390/make7030105`

---

## Prerequisites

- **Python 3.8+** (recommended: Python 3.10 or 3.11)
- **pip** (Python package manager)

---

## Installation & Setup

### Step 1: Clone or download the project

Place all project files in a single folder (e.g., `F:\chunk\`).

### Step 2: Install required packages

Open a terminal/command prompt in the project folder and run:

```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn plotly
```

**Package details:**

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computations |
| `scikit-learn` | ML models (Logistic Regression, Random Forest), preprocessing, metrics |
| `imbalanced-learn` | SMOTE for class imbalance handling |
| `plotly` | Interactive charts and visualizations |

### Step 3: (Optional) Generate sample test data

If you don't have a CSV dataset ready, generate sample data first:

```bash
python generate_test_data.py
```

This creates 4 sample CSV files in the project folder.

---

## Running the Dashboard

```bash
streamlit run churn_dashboard.py
```//python -m streamlit run churn_dashboard.py// (FOR ME TO RUN)


The dashboard will open in your browser at **http://localhost:8501**.

If it doesn't open automatically, manually navigate to that URL.

---

## How to Use

1. **Upload a CSV** — Click the upload button and select any customer churn CSV file.
   - Supported datasets: Telco, Netflix, Banking, SaaS, E-commerce, or any CSV with a churn column.
   - A sample dataset is included: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
2. **Auto-detection** — The system automatically detects the target (churn) column, ID column, and feature types.
3. **View Results** — The dashboard displays:
   - KPI cards (total customers, churn rate, high-risk count)
   - Model comparison (Logistic Regression vs Random Forest)
   - Churn distribution and risk category charts
   - ROC curves
   - Confusion matrices
   - Feature importance rankings
   - Churn breakdown by category
   - High-risk customers table
4. **Download** — Export predictions, model comparison, and feature importance as CSV files.

---

## Project Structure

```
chunk/
├── churn_dashboard.py                  # Main Streamlit dashboard application
├── generate_test_data.py               # Script to generate sample test CSV files
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Sample Telco churn dataset
├── make-07-00105.pdf                   # Reference paper (Imani et al., 2025)
├── kaggel_Customer_Churn_Predictionk.ipynb  # Jupyter notebook for exploration
├── README.md                           # This file
└── check_points/                       # Pre-generated output samples
    ├── churn_predictions.csv
    ├── churn_summary.csv
    ├── feature_importance.csv
    └── model_comparison.csv
```

---

## ML Pipeline (Automated)

The dashboard runs the following pipeline automatically on CSV upload:

1. **Load & Auto-Detect** — Identifies target column, ID column, categorical and numeric features
2. **Preprocessing** — Handles missing values (median/mode), encodes categoricals (LabelEncoder), scales numerics (StandardScaler)
3. **Class Balancing** — Applies SMOTE on training data to handle imbalanced classes
4. **Train-Test Split** — 80/20 stratified split
5. **Model Training** — Trains Logistic Regression and Random Forest classifiers
6. **Evaluation** — Computes Accuracy, Precision, Recall, F1 Score, and AUC-ROC
7. **Risk Scoring** — Categorizes customers into Low / Medium / High risk based on churn probability

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `streamlit` not found | Run `pip install streamlit` |
| `ModuleNotFoundError: imblearn` | Run `pip install imbalanced-learn` |
| Port 8501 already in use | Run `streamlit run churn_dashboard.py --server.port 8502` |
| Dashboard loads but no charts | Make sure you uploaded a CSV file first |
| "No churn column found" | Your CSV doesn't have a column with "churn" in its name — select the target manually from the dropdown |
