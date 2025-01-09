import tensorflow as tf
import pandas as pd
import numpy as np
import streamlit as st

# Load the model and weights
model = tf.keras.models.load_model("Fish_Spech_64.h5")
model.load_weights("Fish_Spech_64.weights.h5")

# Define the input fields
st.title("Fish Species Prediction")
ph = st.number_input("Enter pH", min_value=0.0, max_value=14.0, step=0.1)
temperature = st.number_input("Enter Temperature (°C)", min_value=0.0, max_value=100.0, step=0.1)
turbidity = st.number_input("Enter Turbidity (NTU)", min_value=0.0, max_value=100.0, step=0.1)

# Prepare the input data
list_of_columns = ["ph", "temperature", "turbidity"]
input_data = pd.DataFrame(columns=list_of_columns)
input_data.at[0, "ph"] = (ph - 6) / (8.8 - 6)
input_data.at[0, "temperature"] = (temperature - 4) / (35 - 4)
input_data.at[0, "turbidity"] = (turbidity - 3.56) / (14.8 - 3.56)
data = np.array(input_data[0:])
y_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(y_tensor)
    ind = np.argmax(prediction)
    if ind == 0:
        st.write("Karpio")
    elif ind == 1:
        st.write("Katla")
    elif ind == 2:
        st.write("Shrimp")
    elif ind == 3:
        st.write("Silver")