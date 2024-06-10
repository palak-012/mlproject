import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder



try:
    with open('salarypredictionmodel.pkl', 'rb') as file:
        model = pickle.load(file)

except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    with open('label_encoding.pkl', 'rb') as file:
        label_encode = pickle.load(file)

except Exception as e:
    st.error(f"Error Label encoding columns: {e}")

st.title('Salary Prediction App')

designation = st.selectbox('Enter Designation', ['Entry Level', 'Senior', 'Manager', 'Lead', 'Director'])
age = st.number_input('Enter your Age', min_value=18, max_value=70, value=25)
past_exp = st.number_input('Enter your Past Exp (in years)', min_value=0, max_value=50, value=5)

if st.button('Predict Salary'):

    input_data = pd.DataFrame(np.zeros((1, len(label_encode))), columns=label_encode)

    input_data.at[0, 'AGE'] = age
    input_data.at[0, 'PAST EXP'] = past_exp

    if f'DESIGNATION_{designation}' in input_data.columns:
        input_data.at[0, f'DESIGNATION_{designation}'] = 1

    try:
        salary_prediction = model.predict(input_data)[0]

        st.write(f'Predicted Salary: ${salary_prediction:.2f}')
    except Exception as e:
        st.error(f"An error occurred: {e}")