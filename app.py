import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic_train.csv")
    df = df[['Sex', 'Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Survived']]
    df.dropna(inplace=True)

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    return df

data = load_data()

# Train model
@st.cache_resource
def train_model(data):
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

model = train_model(data)

# Streamlit UI
st.title("Titanic Survival Prediction App ")
st.write("Enter passenger details below to predict survival chances:")

sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
pclass = st.selectbox("Passenger Class (1 = 1st, 3 = 3rd)", [1, 2, 3])
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 6, 0)

sex_encoded = 1 if sex == "Male" else 0
input_data = pd.DataFrame([[sex_encoded, age, fare, pclass, sibsp, parch]],
                          columns=['Sex', 'Age', 'Fare', 'Pclass', 'SibSp', 'Parch'])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("The passenger **survived** ")
    else:
        st.error("The passenger **did not survive** ")
