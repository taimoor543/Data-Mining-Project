import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Function to perform Augmented Dickey-Fuller test
def perform_adf_test(series):
    result = adfuller(series)
    return result

# Load your data
@st.cache
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    return data

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Display the data
    st.write(data)

    # Sidebar for model selection and parameters
    model_option = st.sidebar.selectbox("Select the model to run:", ["ARIMA", "SARIMA", "ETS", "Prophet"])
    value_column_name = st.sidebar.selectbox("Select value column:", ['cycle', 'trend'], index=0)
    date_column_name = 'date'  # This is now fixed since we construct it during data loading

    if st.sidebar.button("Run Model"):
        if model_option == "ARIMA":
            # Assume parameters are preset or add options to modify them
            model = ARIMA(data[value_column_name], order=(1, 1, 1))
            model_fit = model.fit()
            st.write(model_fit.summary())

        elif model_option == "SARIMA":
            model = SARIMAX(data[value_column_name], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_fit = model.fit()
            st.write(model_fit.summary())

        elif model_option == "ETS":
            model = ExponentialSmoothing(data[value_column_name])
            model_fit = model.fit()
            st.write(model_fit.summary())

        elif model_option == "Prophet":
            df = data.rename(columns={date_column_name: 'ds', value_column_name: 'y'})
            prophet_model = Prophet()
            prophet_model.fit(df)
            future = prophet_model.make_future_dataframe(periods=365)
            forecast = prophet_model.predict(future)
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
            fig = prophet_model.plot(forecast)
            st.pyplot(fig)

        # ADF Test output
        adf_result = perform_adf_test(data[value_column_name])
        st.write("ADF Statistic: ", adf_result[0])
        st.write("p-value: ", adf_result[1])
        st.write("Critical Values: ", adf_result[4])