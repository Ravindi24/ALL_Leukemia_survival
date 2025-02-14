import streamlit as st
from PIL import Image
import pickle
import numpy as np
import gzip
import zipfile
import os

# Set background color to light red
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffe6e6;
    }
    .stButton button {
        background-color: #ff4d4d;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton button:hover {
        background-color: #e60000;
        color: white;
    }
    .title {
        color: #e6536e;
        font-weight: bold;
    }
    .output {
        background-color: #f57d7d;
        padding: 10px;
        border-radius: 5px;
    }
    label {
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Decompress .gz file and then extract .zip
gz_file = 'rsf_model.zip.gz'
extracted_zip_file = 'rsf_model.zip'
extracted_model_file = 'rsf_model.pkl'

# Check if the model is already extracted
if not os.path.exists(extracted_model_file):
    # Decompress the .gz file
    with gzip.open(gz_file, 'rb') as f_in:
        with open(extracted_zip_file, 'wb') as f_out:
            f_out.write(f_in.read())
    st.write(f"Decompressed {gz_file} to {extracted_zip_file}")

    # Now extract the .zip file
    with zipfile.ZipFile(extracted_zip_file, 'r') as zip_ref:
        zip_ref.extractall()
    st.write(f"Extracted model from {extracted_zip_file}")

# Load the model
try:
    with open(extracted_model_file, 'rb') as file:
        rsf_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'rsf_model.pkl' not found. Please upload or provide the correct model file.")
    rsf_model = None

def run():
    # Load and display an image
    img1 = Image.open('cancer1.jpeg')
    img1 = img1.resize((800, 150))
    st.image(img1, use_container_width=True)

    
    # App title
    st.markdown("<h1 class='title'>Leukemia Survival Prediction using Machine Learning</h1>", unsafe_allow_html=True)

    ## Clinical No
    Clinical_no = st.text_input('Clinical number')

    ## Gender as radio button
    gen = st.radio("Gender", options=['Female', 'Male'])

    ## Age Slider
    age = st.slider("Age", min_value=0, max_value=19, value=10)

    ## Diagnosis Type
    Diagnosis = st.selectbox("Diagnosis Type", ['B-ALL', 'T-ALL'])

    ## Risk Stratification
    RiskStratification = st.selectbox("Risk Stratification", ['Low Risk', 'High Risk'])

    ## Initial WBC Count
    Initial_WBC = st.text_input('Initial White Blood Cell (WBC) count in μl or mm³')

    # Handle prediction when the button is clicked
    if st.button("Submit"):
        # Validate Initial_WBC input
        if not Initial_WBC:
            st.error("Please enter a value for Initial WBC count.")
            return

        try:
            Initial_WBC = float(Initial_WBC)
        except ValueError:
            st.error("Please enter a valid numerical value for Initial WBC count.")
            return

        # Convert categorical inputs into numeric format for model input
        gen = 1 if gen == 'Male' else 0  # Male -> 1, Female -> 0
        Diagnosis = 1 if Diagnosis == 'T-ALL' else 0  # T-ALL -> 1, B-ALL -> 0
        RiskStratification = 1 if RiskStratification == 'High Risk' else 0  # High Risk -> 1, Low Risk -> 0

        # Create feature list
        features = [[gen, age, Diagnosis, RiskStratification, Initial_WBC]]
        
        # Check if model is loaded
        if rsf_model is None:
            st.error("Model not loaded. Please check the model file.")
            return

        try:
            # Predict survival function
            survival_function = rsf_model.predict_survival_function(features)

            # Extract survival probability at 1 year (365 days)
            time_point = 365*5  # 1 year in days
            survival_probability = None

            # Iterate through survival_function for the first sample
            for fn in survival_function:
                # `fn.x` contains time points, and `fn.y` contains survival probabilities
                survival_probability = float(fn(time_point))  # Evaluate StepFunction at 365 days

            if survival_probability is None:
                st.error("Could not extract survival probability.")
                return

            # Decision based on survival probability
            prediction = 1 if survival_probability > 0.5 else 0  # Threshold for high/low survival
            
            # Display survival probability and prediction
            st.write(f"Predicted Survival Probability at 5 Year: **{survival_probability * 100:.2f}%**")
            output_message = ""
            if prediction == 1:
                output_message = "<div class='output'>Based on our analysis, the survival probability of this patient is high!</div>"
            else:
                output_message = "<div class='output'>Based on our analysis, the survival probability of this patient is low.</div>"
            st.markdown(output_message, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Run the app
if __name__ == '__main__':
    run()

