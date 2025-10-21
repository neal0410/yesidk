
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

X = [[1], [2], [3], [4]] #tipe data list 2 dimensi
y = [101, 102, 103, 104] #tipe data list 1 dimensi
model = LinearRegression()

st.title("Prediksi Gaji")

model.fit(X, y)
input_user = st.number_input("Masukan value :")

prediction = model.predict([[input_user]])

fig, ax = plt.subplots()

prediksi_y = model.predict(X)
ax.scatter(X, y)
ax.plot(X, prediksi_y)
ax.scatter([input_user], [prediction])

st.pyplot(fig)
st.metric(label="Gaji", value=prediction)
