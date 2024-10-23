import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the pre-trained model
model_path = r"C:\Users\hp\ML file\Car Price Prediction\car_model.pkl"
loaded_model = joblib.load(model_path)  # Correct model loading

# Set up Streamlit UI
st.header('Car Price Prediction ML Model')

# Load the car details dataset
car_data = pd.read_csv("Cardetails.csv")

# Function to extract the brand name from the full car name
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip(' ')

car_data['name'] = car_data['name'].apply(get_brand_name)

# Input fields for the user to enter car details
name = st.selectbox('Select Car Brand', car_data['name'].unique())
year = st.slider('Car Manufactured Year', 1993, 2024)
km_Driven = st.slider('Number of Kilometers Driven', 11, 200000)  # Adjusted the upper bound
fuel = st.selectbox('Fuel Type', car_data['fuel'].unique())
seller_type = st.selectbox('Seller Type', car_data['seller_type'].unique())
transmission = st.selectbox('Transmission', car_data['transmission'].unique())
owner = st.selectbox('Owner Type', car_data['owner'].unique())
mileage = st.slider('Car Mileage (in km/l)', 10, 40)
engine = st.slider('Engine Size (in CC)', 700, 5000)
max_power = st.slider('Maximum Power (in HP)', 0, 200)
seats = st.slider('Number of Seats', 5, 10)

# Prediction logic triggered by a button click
if st.button("Predict"):
    # Create a dataframe for the model input
    input_data_model = pd.DataFrame(
        [[name, year, km_Driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    
    # Replace categorical values with numeric values (ensure these mappings match the model training)
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                       'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                      'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                      'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                      'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                      'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                     list(range(1, 32)), inplace=True)
    
    # Make the prediction using the loaded model
    try:
        car_price = loaded_model.predict(input_data_model)
        st.markdown(f'Estimated Car Price: ₹ {car_price[0]:,.2f}')  # Display with currency formatting
    except Exception as e:
        st.error(f"Error in prediction: {e}")
