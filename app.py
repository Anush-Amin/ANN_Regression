import numpy as np
import pandas as pd
import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# Load the model
model = load_model('salary_prediction_model.h5')

# Load label encoder, one hot encoder and standard scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Salary Prediction App')

# User input
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card?', ['Yes', 'No'])
has_credit_card = 1 if has_credit_card == 'Yes' else 0
is_active_member = st.selectbox('Is Active Member?', ['Yes', 'No'])
is_active_member = 1 if is_active_member == 'Yes' else 0
exited = st.selectbox('Exited?', ['Yes', 'No'])
exited = 1 if exited == 'Yes' else 0

# Preprocess the input
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited],
})

# Encode geography
geography_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

# Concatenate the input data and the encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict the salary
prediction = model.predict(input_data_scaled)[0][0]

# Display the prediction
st.write(f'Predicted Salary: {prediction:.2f}') 