import streamlit as st
import pickle
import numpy as np

# Load trained model (assuming your trained model is saved as 'model.pkl')
model_path = 'model.pkl'
# Adjust this path if needed
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app title and description
st.title("House Price Prediction App")
st.write("""
Welcome to the House Price Prediction App!\n
This app predicts Median value of owner-occupied homes in $1000's based on user-provided features.\n
Please fill in the required input parameters below to get a prediction.
""")

# Input fields for user parameters
CRIM = st.number_input("CRIM: Per capita crime rate by town", min_value=0.0, step=0.1, format="%.5f")
ZN = st.number_input("ZN: Proportion of residential land zoned for large lots", min_value=0.0, step=0.1, format="%.2f")
INDUS = st.number_input("INDUS: Proportion of non-retail business acres per town", min_value=0.0, step=0.1, format="%.2f")
CHAS = st.selectbox("CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)", [0, 1])
NOX = st.number_input("NOX: Nitric oxides concentration (parts per 10 million)", min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
RM = st.number_input("RM: Average number of rooms per dwelling", min_value=0.0, step=0.1, format="%.2f")
AGE = st.number_input("AGE: Proportion of owner-occupied units built prior to 1940", min_value=0.0, max_value=100.0, step=0.1, format="%.2f")
DIS = st.number_input("DIS: Weighted distances to five Boston employment centers", min_value=0.0, step=0.1, format="%.2f")
RAD = st.number_input("RAD: Index of accessibility to radial highways", min_value=1, step=1)
TAX = st.number_input("TAX: Full-value property-tax rate per $10,000", min_value=0, step=1)
PTRATIO = st.number_input("PTRATIO: Pupil-teacher ratio by town", min_value=0.0, step=0.1, format="%.2f")
B = st.number_input("B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town", min_value=0.0, step=0.1, format="%.2f")
LSTAT = st.number_input("LSTAT: % lower status of the population", min_value=0.0, step=0.1, format="%.2f")

# Make prediction
if st.button("Predict"):
    # Collect input data into a single numpy array
    input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

    # Ensure that the input data is the correct shape (1, number_of_features)
    input_data = input_data.reshape(1, -1)
    st.write(f"Input data shape: {input_data.shape}")  # Print the shape to verify

    # Make prediction using the loaded model
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
        
        # Display the user input summary
        st.write("### Summary of Your Inputs:")
        st.write(f"**CRIM**: {CRIM}")
        st.write(f"**ZN**: {ZN}")
        st.write(f"**INDUS**: {INDUS}")
        st.write(f"**CHAS**: {CHAS}")
        st.write(f"**NOX**: {NOX}")
        st.write(f"**RM**: {RM}")
        st.write(f"**AGE**: {AGE}")
        st.write(f"**DIS**: {DIS}")
        st.write(f"**RAD**: {RAD}")
        st.write(f"**TAX**: {TAX}")
        st.write(f"**PTRATIO**: {PTRATIO}")
        st.write(f"**B**: {B}")
        st.write(f"**LSTAT**: {LSTAT}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")