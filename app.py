import streamlit as st

import joblib as jl
import pandas as pd

st.set_page_config(page_title='Mid Test', layout='wide')

st.title('Heart Disease Classification ❤️‍🩹')
st.header('Using Random Forest Classifier from Scikit-Learn', divider='red')

margin_left, col_left, margin_mid, col_right, margin_right = st.columns([0.2,5,1,5,0.2])

age = col_right.slider('How old are you?', 0, 100, 50)

sex = col_left.selectbox(
    'Select Sex',
    ('Male', 'Female'),
    index=None,
    placeholder='Sex'
)
if sex == 'Male':
    sex = 1
elif sex == 'Female':
    sex = 0

cp = col_left.selectbox(
    'Select Chest Pain Type',
    ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'),
    index=None,
    placeholder='Chest Pain Type'
)
if cp == 'Typical Angina':
    cp = 1
elif cp == 'Atypical Angina':
    cp = 2
elif cp == 'Non-Anginal Pain':
    cp = 3
elif cp == 'Asymptomatic':
    cp = 4

trestbps = col_right.slider('Resting Blood Pressure', 0, 300, 150)

restecg = col_left.selectbox(
    'Select Resting Electrocardiographic Results',
    ('Normal', 'Having ST-T Wave Abnormality', 'Probable or Definite Left Ventricular Hypertrophy'),
    index=None,
    placeholder='Resting Electrocardiographic Results'
)
if restecg == 'Normal':
    restecg = 0
elif restecg == 'Having ST-T Wave Abnormality':
    restecg = 1
elif restecg == 'Probable or Definite Left Ventricular Hypertrophy':
    restecg = 2

thalach = col_right.slider('Maximum Heart Rate Achieved', 0, 300, 150)

exang = col_left.selectbox(
    'Does Exercise Induced Angina?',
    ('Yes', 'No'),
    index=None,
    placeholder='Exercise Induced Angina'
)
if exang == 'Yes':
    exang = 1
elif exang == 'No':
    exang = 0

oldpeak = col_right.slider('ST Depression Induced by Exercise Relative to Rest', 0, 5, 3)

if 'result' not in st.session_state:
    st.session_state.result = 'Fill the column and click Predict Button.'

def predict():
    x = {
        'age' : [age],
        'sex' : [sex],
        'cp' : [cp],
        'trestbps' : [trestbps],
        'restecg' : [restecg],
        'thalach' : [thalach],
        'exang' : [exang],
        'oldpeak' : [oldpeak],
    }
    x = pd.DataFrame(x)

    model = jl.load('RMC_heart_disease_Classification.joblib')
    
    try:
        pred = model.predict(x)
        pred = pred[0]

        if pred == 0:
            st.session_state.result = "The result is : \< 50% Diameter Narrowing."
        elif pred == 1:
            st.session_state.result = "The result is : \> 50% Diameter Narrowing."
        else:
            st.error('Something Wrong')
    except:
        st.session_state.result = 'Something Wrong.'
        

col_left.button('Predict', on_click=predict)

col_left.write(st.session_state.result)
