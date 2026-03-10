import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)
with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)
    
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography',one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products', 0, 10)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_memeber = st.selectbox("Is active memeber", [0,1])


input_data = pd.DataFrame({
    'CreditScore':[credit_score], 
    'Geography': [geography], 
    'Gender': [gender], 
    'Age': [age], 
    'Tenure': [tenure], 
    'Balance': [balance], 
    'NumOfProducts':[num_of_products], 
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_memeber],
    'EstimatedSalary':[estimated_salary]
})

# encode gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# encode geography
geo_encoded = one_hot_encoder.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder.get_feature_names_out(['Geography'])
)

# drop original geography and add encoded columns
input_data = pd.concat(
    [input_data.drop('Geography', axis=1).reset_index(drop=True), geo_encoded_df],
    axis=1
)

# scale
input_data_scaled = scalar.transform(input_data)
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn probability: {prediction_proba:.2f}')
if prediction_proba > 0.5:
    st.write('The Customer is Likely to CHURN')
else: 
    st.write("The Customer won't CHURN")