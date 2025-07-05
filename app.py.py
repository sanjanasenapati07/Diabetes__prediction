# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 21:01:12 2025

@author: Lenovo
"""

import numpy as np
import pickle 
import streamlit as st

loaded_model = pickle.load(open('C:/Users/Lenovo/Desktop/Diabetes_Prediction/trained_model.sav','rb'))

#creating function for prediction
def diabetes_prediction(ip):
    ip_as_numarray = np.asarray(ip)
    ip_reshaped = ip_as_numarray.reshape(1,-1)

    prediction = loaded_model.predict(ip_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return "The person is not diabetic"
    else:
      return "The person is diabetic"
  
def main():
    
    st.title('Diabetes Prediction Web App')
    
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')
    
    diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
if __name__ =='__main__':
    main()