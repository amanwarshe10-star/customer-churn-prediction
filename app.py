import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from preprocess import prepare_data

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("📊 Customer Churn Predictor")
st.write("Fill in customer details to predict if they will churn.")
st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.0)
    contract = st.selectbox("Contract Type", 
                            ['Month-to-month', 'One year', 'Two year'])

with col2:
    internet = st.selectbox("Internet Service", 
                            ['DSL', 'Fiber optic', 'No'])
    tech_support = st.selectbox("Tech Support", ['Yes', 'No'])
    senior = st.selectbox("Senior Citizen", ['No', 'Yes'])

if st.button("🔮 Predict Churn", type="primary"):
    
    contract_val = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[contract]
    internet_val = {'DSL': 0, 'Fiber optic': 1, 'No': 2}[internet]
    tech_val = {'No': 0, 'Yes': 1}[tech_support]
    senior_val = {'No': 0, 'Yes': 1}[senior]
    total_charges = tenure * monthly_charges

    inp = np.zeros((1, 19))
    inp[0][0] = senior_val
    inp[0][4] = tenure
    inp[0][5] = contract_val
    inp[0][10] = tech_val
    inp[0][14] = internet_val
    inp[0][17] = monthly_charges
    inp[0][18] = total_charges

    prob = model.predict_proba(inp)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    col_a, col_b = st.columns(2)
    with col_a:
        if prob > 0.5:
            st.error(f"⚠️ WILL CHURN")
        else:
            st.success(f"✅ WILL NOT CHURN")
    with col_b:
        st.metric("Churn Probability", f"{prob:.1%}")

    st.progress(float(prob))

    st.subheader("💡 Recommended Action")
    if prob > 0.6:
        st.error("High Risk! Offer discount or personal callback immediately.")
    elif prob > 0.3:
        st.warning("Medium Risk. Send retention email with special offer.")
    else:
        st.success("Low Risk. Customer is satisfied.")