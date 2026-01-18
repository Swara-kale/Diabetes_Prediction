# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 12:49:12 2026

@author: swarangi kale
"""

import numpy as np
import pickle
import streamlit as st
import base64

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
loaded_model = pickle.load(open("trained_model.sav", 'rb'))

# ---------------- PREDICTION FUNCTION ----------------
def diabetes_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "The person is NOT diabetic"
    else:
        return "The person IS diabetic"
    
def add_bg_image(image_file):
    # Step 1: Open the image in binary mode
    with open(image_file, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()

    # Step 2: Apply CSS for background + blur
    st.markdown(
          f"""
         <style>

    /* ===== FULL APP BACKGROUND ===== */
    .stApp {{
        background-image:
        linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.55)),
        url("data:image/jpeg;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* ===== GLASS / BLUR EFFECT FOR CONTENT ===== */
    section[data-testid="stVerticalBlock"] {{
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        background-color: rgba(30, 90, 150, 0.35);
        border-radius: 22px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }}

    /* ===== INPUT BOX STYLING ===== */
    input, textarea {{
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }}

    /* ===== RESULT CARD ===== */
    .result-box {{
        margin-top: 20px;
        padding: 18px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
    }}

    .result-safe {{
        background-color: rgba(20, 140, 80, 0.9);
        color: white;
    }}

    .result-danger {{
        background-color: rgba(180, 40, 40, 0.9);
        color: white;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

 
        
# ---------------- MAIN APP FUNCTION ----------------
def main():
    add_bg_image("bg_diabetes.jpeg")
   
    # Title
    st.title("🩺 Diabetes Prediction System")

    # Description
    st.markdown("""
    This web application predicts whether a person is diabetic  
    using a trained Machine Learning classification model.
    """)

   

    # Layout using columns
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0)
        Glucose = st.number_input("Glucose Level")
        BloodPressure = st.number_input("Blood Pressure")
        SkinThickness = st.number_input("Skin Thickness")

    with col2:
        Insulin = st.number_input("Insulin Level")
        BMI = st.number_input("BMI")
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function")
        Age = st.number_input("Age", min_value=1)

    # Prediction button
    if st.button("🔍 Predict Diabetes"):
        diagnosis = diabetes_prediction([
            Pregnancies, Glucose, BloodPressure,
            SkinThickness, Insulin, BMI,
            DiabetesPedigreeFunction, Age
        ])

        if diagnosis == "The person is NOT diabetic":
         st.markdown(
        '<div class="result-box result-safe">🟢 The person is NOT diabetic</div>',
        unsafe_allow_html=True
         )
        else:
         st.markdown(
        '<div class="result-box result-danger">🔴 The person IS diabetic</div>',
        unsafe_allow_html=True
    )

    # Extra explanation
    with st.expander("ℹ️ How does this app work?"):
        st.write("""
        • The user enters basic medical details such as glucose level, BMI, and age.

• These inputs are sent to a trained Machine Learning classification model.

• The model analyzes the patterns based on historical medical data.

• Based on the analysis, the model predicts whether the person is diabetic or not.

• The prediction result is instantly displayed on the screen for the user..
        """)

# ---------------- EXECUTION ----------------
if __name__ == "__main__":
    main()
