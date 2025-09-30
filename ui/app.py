import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

pipeline = joblib.load("../models/final_model.pkl")
df = pd.read_csv("../data/heart_disease.csv")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient health information below to predict the likelihood of heart disease.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)

with col2:
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.5, value=1.0, step=0.1)
    cp_4 = st.selectbox("Chest Pain Type (cp_4)", [0, 1], format_func=lambda x: "Type 0" if x == 0 else "Type 1")
    thal_7 = st.selectbox("Thalassemia (thal_7)", [0, 1], format_func=lambda x: "Normal" if x == 0 else "Fixed Defect")
    exang = st.selectbox("Exercise-Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

if st.button("Predict Heart Disease Risk"):
    input_data = pd.DataFrame([{
        "oldpeak": oldpeak,
        "cp_4": int(cp_4),
        "thal_7": int(thal_7),
        "exang": int(exang),
        "thalach": thalach,
        "chol": chol,
        "age": age,
        "trestbps": trestbps
    }])
    
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"🚨 High Risk of Heart Disease (Probability: {probability:.1%})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {probability:.1%})")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50}}
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")
st.markdown("*Use this tool to understand heart disease risk factors and make informed healthcare decisions.*")
