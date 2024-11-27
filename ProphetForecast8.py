import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

# Setup Streamlit
st.title("Organic Traffic Prediction using Prophet")

# Sidebar inputs
st.sidebar.header("User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload 'Ahrefs Traffic Data' CSV file", type=["csv"])

# Optional parameters
weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=True)
alg_updates = st.sidebar.checkbox("Algorithm Updates")

referring_domains_check = st.sidebar.checkbox("Referring Domains", value=True)
changepoint_range = st.sidebar.text_input("Changepoint Range", "0.9")
changepoint_prior_scale = st.sidebar.text_input("Trend Flexibility", "0.05")

# Algorithm update dates
alg_update_dates = [
    '2024-11-11', '2024-08-15', '2024-06-20', '2024-05-14', '2024-05-06', '2024-03-05',
    '2023-11-08', '2023-11-02', '2023-10-05', '2023-10-04', '2023-09-14', '2023-08-22',
    '2023-04-12', '2023-03-15', '2023-02-21', '2022-12-14', '2022-12-05', '2022-10-19',
    '2022-09-20', '2022-09-12'
]

if uploaded_file is not None:
    # Load data, skip the second and third lines
    df = pd.read_csv(uploaded_file, skiprows=[1, 2])
    
    # Extract necessary columns
    df = df[['Metric', ' Referring domains', ' Organic traffic']]
    df.columns = ['ds', 'referring_domains', 'y']
    
    # Split data into forecast_data and test_data
    df['ds'] = pd.to_datetime(df['ds'])
    df.sort_values('ds', inplace=True)
    
    test_data = df.iloc[-182:]
    forecast_data = df.iloc[:-182]

    # Create a Prophet model
    m = Prophet(daily_seasonality=False, weekly_seasonality=weekly_seasonality,
                changepoint_range=float(changepoint_range),
                changepoint_prior_scale=float(changepoint_prior_scale))
    
    # Add extra regressors if ticked
    if referring_domains_check:
        m.add_regressor('referring_domains')
    
    if alg_updates:
        holidays = pd.DataFrame({
            'holiday': 'alg_update',
            'ds': pd.to_datetime(alg_update_dates),
            'lower_window': 0,
            'upper_window': 1,
        })
        m.add_country_holidays(country_name='US')  # Assuming US holidays, adjust as needed
        m = Prophet(holidays=holidays)
    
    # Fit the model
    m.fit(forecast_data)
    
    # Make future dataframe
    future = m.make_future_dataframe(periods=182)
    if referring_domains_check:
        # Extend referring_domains to future dates (using mean as a simplistic approach)
        referring_domains_future = pd.Series([forecast_data['referring_domains'].mean()] * len(future), index=future.index)
        future['referring_domains'] = referring_domains_future
    
    # Predict
    forecast = m.predict(future)

    # MAE for each day
    y_true = test_data['y'].values
    y_pred = forecast.iloc[-182:]['yhat'].values
    mae_per_day = mean_absolute_percentage_error(y_true, y_pred)

    # Display MAE per day in table
    test_data['MAPE'] = abs((test_data['y'] - forecast.iloc[-182:]['yhat']) / test_data['y']) * 100
    st.subheader("Mean Absolute Percentage Error (MAPE) Per Day")
    st.table(test_data[['ds', 'MAPE']])

    # Weighted Average Percentage Error
    weighted_mape = (test_data['y'] * test_data['MAPE']).sum() / test_data['y'].sum()
    st.subheader("Weighted Average Percentage Error")
    st.text(weighted_mape)

    # Plot the results
    st.subheader("Predicted vs Actual Organic Traffic")
    fig = plot_plotly(m, forecast)
    fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data['y'], mode='markers', name='Actual'))
    
    st.plotly_chart(fig)

else:
    st.info("Please upload a valid CSV file.")