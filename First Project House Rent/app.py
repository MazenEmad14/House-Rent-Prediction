import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, encoder, and scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("🏠 House Rent Prediction App")

# --- Inputs from user ---
BHK = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)
Size = st.number_input("Size (in sqft)", min_value=100)
Area_Type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
Furnishing_Status = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
Bathroom = st.slider("Number of Bathrooms", min_value=1, max_value=10)
Month = st.selectbox("Posted Month", list(range(1, 13)))
Floor_Num = st.number_input("Floor Number", min_value=-2)
Apartment_num = st.number_input("Apartments in Building", min_value=1)

City = st.selectbox("City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
Tenant = st.selectbox("Tenant Preferred", ["Bachelors", "Bachelors/Family", "Family"])
Contact = st.selectbox("Point of Contact", ["Contact Agent", "Contact Builder", "Contact Owner"])

# --- Data Preparation ---
# Label Encode
area_type_map = {"Super Area": 2, "Carpet Area": 0, "Built Area": 1}
furnishing_map = {"Furnished": 1, "Semi-Furnished": 2, "Unfurnished": 0}

Area_Type_enc = area_type_map[Area_Type]
Furnishing_Status_enc = furnishing_map[Furnishing_Status]


# OneHot Encode (manually)
city_cols = ['City_Bangalore', 'City_Chennai', 'City_Delhi', 'City_Hyderabad', 'City_Kolkata', 'City_Mumbai']
tenant_cols = ['Tenant Preferred_Bachelors', 'Tenant Preferred_Bachelors/Family', 'Tenant Preferred_Family']
contact_cols = ['Point of Contact_Contact Agent', 'Point of Contact_Contact Builder', 'Point of Contact_Contact Owner']

city_data = [1 if f"City_{City}" == col else 0 for col in city_cols]
tenant_data = [1 if f"Tenant Preferred_{Tenant}" == col else 0 for col in tenant_cols]
contact_data = [1 if f"Point of Contact_{Contact}" == col else 0 for col in contact_cols]

# Combine all features
input_data = pd.DataFrame([[
    BHK, Size, Area_Type_enc, Furnishing_Status_enc,
    Bathroom, Month, Floor_Num, Apartment_num
] + city_data + tenant_data + contact_data],
columns=[
    'BHK', 'Size', 'Area Type', 'Furnishing Status',
    'Bathroom', 'Month', 'Floor_Num', 'Apartment_num'
] + city_cols + tenant_cols + contact_cols)

# Scale numerical features
scale_cols = ['BHK', 'Size', 'Bathroom', 'Month', 'Floor_Num', 'Apartment_num']
input_data[scale_cols] = scaler.transform(input_data[scale_cols])

# --- Prediction ---
if st.button("Predict Rent"):
    prediction = model.predict(input_data)
    st.success(f"🏡 Estimated Rent: ₹ {int(prediction[0]):,}")
