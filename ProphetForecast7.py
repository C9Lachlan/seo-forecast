import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go  # Import Plotly's graph_objects module

st.title('Organic Traffic Forecast with Prophet')

# Sidebar for user inputs
uploaded_file = st.sidebar.file_uploader("Upload Ahrefs Traffic Data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, skiprows=[1, 2])
    df = df[['Metric', ' Referring domains', ' Organic traffic']]
    df.columns = ['ds', 'referring_domains', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    # Split into training and testing datasets
    forecast_data = df.iloc[:-182]
    test_data = df.iloc[-182:]

    # User inputs
    weekly_seasonality = st.sidebar.checkbox('Weekly Seasonality', value=True)
    algorithm_updates = st.sidebar.checkbox('Algorithm Updates')
    referring_domains = st.sidebar.checkbox('Referring Domains')
    changepoint_range = st.sidebar.text_input('Changepoint Range', '0.9')
    trend_flexibility = st.sidebar.text_input('Trend Flexibility', '0.05')

    # Initialize Prophet model
    m = Prophet(
        weekly_seasonality=weekly_seasonality,
        changepoint_range=float(changepoint_range),
        changepoint_prior_scale=float(trend_flexibility),
    )

    # Add regressors based on user input
    if algorithm_updates:
        algorithm_dates = [
            '2024-11-11', '2024-08-15', '2024-06-20', '2024-05-14', '2024-05-06',
            '2024-03-05', '2023-11-08', '2023-11-02', '2023-10-05', '2023-10-04',
            '2023-09-14', '2023-08-22', '2023-04-12', '2023-03-15', '2023-02-21',
            '2022-12-14', '2022-12-05', '2022-10-19', '2022-09-20', '2022-09-12'
        ]
        for date in algorithm_dates:
            forecast_data[date] = forecast_data['ds'].apply(lambda x: 1 if x == pd.Timestamp(date) else 0)
            m.add_regressor(date)

    if referring_domains:
        m.add_regressor('referring_domains')

    # Fit model
    m.fit(forecast_data)

    # Create future dataframe
    future = m.make_future_dataframe(periods=182)
    if referring_domains:
        future = future.merge(forecast_data[['ds', 'referring_domains']], how='left', on='ds')

    # Make forecast
    forecast = m.predict(future)

    # Evaluate model performance
    mape_scores = [
        mean_absolute_percentage_error([test_data.iloc[i]['y']], [forecast.iloc[i - len(forecast_data)]['yhat']])
        for i in range(len(test_data))
    ]
    weighted_mape = sum([mape * test_data.iloc[i]['y'] for i, mape in enumerate(mape_scores)]) / sum(test_data['y'])

    # Display results
    st.subheader('Forecast Results')
    st.write('Mean Absolute Percentage Error per Day:')
    st.write(pd.DataFrame({'Date': test_data['ds'], 'MAPE': mape_scores}))
    
    st.write('Weighted Average Percentage Error:')
    st.write(weighted_mape)

    # Plot the forecast
    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig)