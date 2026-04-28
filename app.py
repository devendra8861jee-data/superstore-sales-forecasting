import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("sales_forecasting_model.pkl")

st.title("Sales Forecasting App")

st.write("Enter details to predict future sales")

year = st.number_input("Year", min_value=2020, max_value=2030, value=2024)
month = st.number_input("Month", min_value=1, max_value=12, value=5)
day = st.number_input("Day", min_value=1, max_value=31, value=15)
quantity = st.number_input("Quantity", min_value=1, value=3)
discount = st.number_input("Discount", min_value=0.0, max_value=1.0, value=0.2)
profit = st.number_input("Profit", value=150.0)

if st.button("Predict Sales"):
    input_data = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Quantity': [quantity],
        'Discount': [discount],
        'Profit': [profit]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: ₹ {prediction[0]:.2f}")

    report = pd.DataFrame({
        "Year": [year],
        "Month": [month],
        "Predicted Sales": [prediction[0]]
    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Report",
        data=csv,
        file_name="sales_report.csv",
        mime="text/csv"
    )
import matplotlib.pyplot as plt

if st.button("Show Sales Graph"):
    fig, ax = plt.subplots(figsize=(10,5))

    actual_sales = [500, 600, 700, 800, 750]
    predicted_sales = [520, 610, 690, 790, 770]

    ax.plot(actual_sales, label="Actual Sales")
    ax.plot(predicted_sales, label="Predicted Sales")

    ax.set_title("Actual vs Predicted Sales")
    ax.legend()

    st.pyplot(fig)
