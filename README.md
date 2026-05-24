Customer Churn Prediction 

Problem Statement
Telecom companies lose revenue when customers leave (churn).
This project predicts which customers are likely to churn so the company can take action before it happens.

Dataset
- Source: Telco Customer Churn (Kaggle)
- 7,043 customers, 21 features

Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- SHAP (Explainability)
- Streamlit (Web App)

Project Structure
- eda.py - Exploratory Data Analysis
- preprocess.py - Data Cleaning and Preprocessing
- train.py - Model Training
- explain.py - SHAP Explainability
- app.py - Streamlit Web App

Results
| Metric | Score |
|--------|-------|
| Accuracy | 77% |
| ROC-AUC | 0.85 |
| Precision (Churn) | 0.55 |
| Recall (Churn) | 0.71 |

Key Findings
- Contract type is the most important feature
- Month-to-month customers churn the most
- New customers (low tenure) have higher churn risk
- High monthly charges increase churn probability

How to Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Author
Aman Warshe
