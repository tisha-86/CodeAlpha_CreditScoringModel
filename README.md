# CodeAlpha_CreditScoringModel

## 📌 Objective
Predict an individual's creditworthiness (likelihood of loan default) using past financial and demographic data. This project is submitted as **Task 1: Credit Scoring Model** for the CodeAlpha Machine Learning Internship.

## 📊 Dataset
- **Source:** Credit Risk Dataset (public, ~32,581 loan applicant records)
- **Target variable:** `loan_status` (1 = default, 0 = repaid)

**Features used:**
`person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`

## ⚙️ Approach
1. **Data Cleaning** — removed data-entry errors (unrealistic ages/employment lengths), imputed missing values with the median.
2. **Exploratory Data Analysis** — correlation heatmap, class balance check.
3. **Feature Engineering** — one-hot encoded categorical features (home ownership, loan intent, loan grade, default history).
4. **Preprocessing** — train/test split (80/20, stratified), feature scaling with `StandardScaler`.
5. **Model Training** — compared three classification algorithms:
   - Logistic Regression
   - Decision Tree
   - Random Forest
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC for each model.
7. **Visualization** — ROC curves, confusion matrix, and feature importance (Random Forest).

## 🏆 Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Random Forest        | 0.93     | 0.96      | 0.71   | 0.82     | **0.92**|
| Decision Tree         | 0.93     | 0.94      | 0.70   | 0.80     | 0.90    |
| Logistic Regression   | 0.87     | 0.78      | 0.56   | 0.65     | 0.87    |

**Best model: Random Forest**, with the strongest ROC-AUC (0.92) and highest precision (0.96) — meaning very few "safe" applicants get wrongly flagged as high-risk.

## 📁 Files in this repo
- `credit_scoring.py` — full training & evaluation script
- `credit_risk_dataset.csv` — dataset used
- `model_comparison.csv` — metrics table for all models
- `correlation_heatmap.png` — feature correlation heatmap
- `target_distribution.png` — class balance plot
- `roc_curves.png` — ROC curve comparison across models
- `confusion_matrix_best_model.png` — confusion matrix for the best model
- `feature_importance.png` — Random Forest feature importance

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python credit_scoring.py
```

## 🔧 Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## 🎓 Internship
This project was completed as part of the **CodeAlpha Machine Learning Internship**.
🔗 [www.codealpha.tech](https://www.codealpha.tech)
