import streamlit as st
import pandas as pd
import numpy as np  
import pickle

st.title("Iris Flower Prediction....")
st.write("This is a simple web app to predict the species of iris flower based on its features.")

sl = st.number_input("Sepal Length (cm)")
      
sw = st.number_input("Sepal Width (cm)")

pl = st.number_input("Petal Length (cm)")

pw = st.number_input("Petal Width (cm)")
button = st.button("Submit")
if button:
   
    # Load the model
    dt= pickle.load(open("iris.pkl", "rb"))
    res=dt.predict([[sl, sw, pl, pw]])[0]
    st.markdown(f"The predicted species is: {res}")

    
   
