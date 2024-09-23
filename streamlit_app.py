# Import necessary libraries
import streamlit as st
import joblib
import pandas as pd

# Load the saved stacking model pipeline
stacking_pipeline = joblib.load('stacking_pipeline.pkl')

# Title of the Streamlit app
st.title("Car Price Prediction App")

# App description
st.write("""
This app predicts the **Selling Price** of a used car based on its features. Please fill in the car details below:
""")

# --- User inputs for prediction ---
# Present Price
present_price = st.number_input('Present Price of the Car (in Lakhs)', min_value=0.0, step=0.1)

# KMs Driven
driven_kms = st.number_input('Driven KMs', min_value=0.0, step=1.0)

# Owner type (0: First Owner, 1: Second Owner, etc.)
owner = st.selectbox('Owner Type', options=[0, 1, 2, 3], format_func=lambda x: f"{x} Owner")

# Year of the car
year = st.slider('Year of Purchase', min_value=1990, max_value=2024, value=2015)

# Fuel Type (Categorical)
fuel_type = st.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'CNG'])

# Selling Type (Categorical)
selling_type = st.selectbox('Selling Type', options=['Individual', 'Dealer'])

# Transmission (Categorical)
transmission = st.selectbox('Transmission Type', options=['Manual', 'Automatic'])

# Button for prediction
if st.button('Predict Selling Price'):
    
    # Creating a new data point for prediction
    input_data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Driven_kms': [driven_kms],
        'Owner': [owner],
        'Fuel_Type': [fuel_type],
        'Selling_type': [selling_type],
        'Transmission': [transmission]
    })

    # Predict using the saved stacking model pipeline
    predicted_price = stacking_pipeline.predict(input_data)

    # Display the predicted selling price
    st.success(f"The predicted selling price for the car is: ₹{predicted_price[0]:.2f} Lakhs")
