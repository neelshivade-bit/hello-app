import streamlit as st
import numpy as np
import pickle

# Load the trained logistic regression model
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title(" Titanic Survival Predictor")

# Sidebar for user inputs
st.sidebar.header("Passenger Information")

# Input fields
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 30)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 5, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 5, 0)
fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 50.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical variables
sex_encoded = 1 if sex == "female" else 0
embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]

# Prepare input for prediction
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    result = " Survived" if prediction == 1 else " Did Not Survive"
    st.subheader("Prediction Result:")
    st.success(result)
