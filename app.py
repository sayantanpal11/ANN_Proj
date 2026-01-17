import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# ---- Load model ----
model = tf.keras.models.load_model('mymodel.h5')

# ---- Load encoders & scaler ----
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---- UI ----
st.title('Customer Churn Prediction')

geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=600.0)
estimated_salary = st.number_input('Estimated Salary', value=50000.0)
tenure = st.slider('Tenure', 0, 10)
num_products = st.slider('Number of Products', 1, 4)
has_cr = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# ---- Encode Gender ----
gender_encoded = label_encoder_gender.transform([gender])[0]

# ---- Base numeric input ----
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cr],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# ---- Encode Geography ----
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)

# ---- Merge ----
input_df = pd.concat([input_df, geo_df], axis=1)

# ---- 🔥 CRITICAL ALIGNMENT FIX ----
# 1. Add missing columns
for col in scaler.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

# 2. Remove extra columns
input_df = input_df[scaler.feature_names_in_]

# ---- Scale ----
input_scaled = scaler.transform(input_df)

# ---- Predict ----
prediction = model.predict(input_scaled)
prob = float(prediction[0][0])

# ---- Output ----
st.write(f"Churn Probability: {prob:.2f}")

if prob > 0.5:
    st.error("Customer is likely to churn")
else:
    st.success("Customer is unlikely to churn")
