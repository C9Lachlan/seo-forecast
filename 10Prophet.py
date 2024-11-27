import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

# Streamlit setup
st.title("Organic Traffic Prediction using Prophet")

# Sidebar inputs
st.sidebar.header("User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload 'Ahrefs Traffic Data' CSV file", type=["csv"])

# Optional parameters
weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=True)
yearly_seasonality = st.sidebar.checkbox("Yearly Seasonality", value=False)
monthly_seasonality = st.sidebar.checkbox("Monthly Seasonality", value=False)
alg_updates = st.sidebar.checkbox("Algorithm Updates")
referring_domains_check = st.sidebar.checkbox("Referring Domains", value=True)
outlier_adjustment = st.sidebar.checkbox("Outlier Detection and Adjustment")

seasonality_mode = st.sidebar.selectbox("Seasonality Mode", options=["additive", "multiplicative"])
changepoint_range = float(st.sidebar.text_input("Changepoint Range", "0.9"))
changepoint_prior_scale = float(st.sidebar.number_input("Changepoint Scale", min_value=0.01, max_value=10.0, value=0.05, step=0.01))
seasonality_prior_scale = float(st.sidebar.number_input("Seasonality Prior Scale", min_value=0.01, max_value=10.0, value=10.0, step=0.1))

# Cap and Floor for logistic growth
use_logistic_growth = st.sidebar.checkbox("Use Logistic Growth")
cap = float(st.sidebar.number_input("Cap", min_value=0.0, step=0.1)) if use_logistic_growth else None
floor = float(st.sidebar.number_input("Floor", min_value=0.0, step=0.1)) if use_logistic_growth else None

# Algorithm update dates
alg_update_dates = [
    '2024-11-11', '2024-08-15', '2024-06-20', '2024-05-14', '2024-05-06', '2024-03-05',
    '2023-11-08', '2023-11-02', '2023-10-05', '2023-10-04', '2023-09-14', '2023-08-22',
    '2023-04-12', '2023-03-15', '2023-02-21', '2022-12-14', '2022-12-05', '2022-10-19',
    '2022-09-20', '2022-09-12'
]

# Process the uploaded data
if uploaded_file is not None:
    # Load data, skip the second and third lines
    df = pd.read_csv(uploaded_file, skiprows=[1, 2])
    
    # Extract necessary columns
    df = df[['Metric', ' Referring domains', ' Organic traffic']]
    df.columns = ['ds', 'referring_domains', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Split data into forecast_data and test_data
    df.sort_values('ds', inplace=True)
    test_data = df.iloc[-182:]
    forecast_data = df.iloc[:-182]

    # Create a Prophet model with specified parameters
    growth = "logistic" if use_logistic_growth else "linear"
    m = Prophet(
        growth=growth,
        daily_seasonality=False,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_mode=seasonality_mode,
        changepoint_range=changepoint_range,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    
    if monthly_seasonality:
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    if referring_domains_check:
        m.add_regressor('referring_domains')
    
    if alg_updates:
        holidays = pd.DataFrame({
            'holiday': 'alg_update',
            'ds': pd.to_datetime(alg_update_dates),
            'lower_window': 0,
            'upper_window': 1,
        })
        m.add_country_holidays(country_name='AU')  # Assuming US holidays; adjust as necessary
        m = Prophet(holidays=holidays)

    # Adjust for outliers or employ preprocessing if necessary
    if outlier_adjustment:
        # Implement outlier detection/adjustment logic if available
        pass

    # Set cap and floor for logistic growth
    if use_logistic_growth:
        forecast_data['cap'] = cap
        forecast_data['floor'] = floor

    # Fit the model
    m.fit(forecast_data)
    
    # Make future dataframe
    future = m.make_future_dataframe(periods=182)
    if referring_domains_check:
        future['referring_domains'] = forecast_data['referring_domains'].mean()
    
    # Set cap and floor for logistic growth in the future dataframe
    if use_logistic_growth:
        future['cap'] = cap
        future['floor'] = floor
    
    # Predict
    forecast = m.predict(future)

    # Calculate MAE for each day
    y_true = test_data['y'].values
    y_pred = forecast.iloc[-182:]['yhat'].values
    mae_per_day = mean_absolute_percentage_error(y_true, y_pred)

    # Display MAE per day in table
    test_data['MAPE'] = abs((test_data['y'] - forecast.iloc[-182:]['yhat']) / test_data['y']) * 100
    #st.subheader("Mean Absolute Percentage Error (MAPE) Per Day")
    #st.table(test_data[['ds', 'MAPE']])

    # Weighted Average Percentage Error
    weighted_mape = (test_data['y'] * test_data['MAPE']).sum() / test_data['y'].sum()
    st.subheader("Weighted Average Percentage Error")
    st.text(weighted_mape)

    # Plot the results
    st.subheader("Predicted vs Actual Organic Traffic")
    fig = plot_plotly(m, forecast)

    # Adjust the layout to start Y-axis at 0 and define a brighter color for the forecast
    fig.update_layout(
        yaxis=dict(range=[0, max(forecast['yhat'].max(), test_data['y'].max())]),  # Ensure y starts at 0
        template="plotly"
    )

    # Customize the forecast line and actual data points color
    # Add a scatter for yhat (forecast) making sure it's visible
    forecast_line = go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='skyblue', width=2)  # Bright color for forecast line
    )
    # Scatter for actual test data
    actual_test_line = go.Scatter(
        x=test_data['ds'],
        y=test_data['y'],
        mode='markers',
        name='Actual Test',
        marker=dict(
            color='lightgreen',  # Slightly darker color for actual test scatter
            size=2
        )
    )
    # Scatter for actual forecast data
    actual_forecast_line = go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['y'],
        mode='markers',
        name='Actual Forecast',
        marker=dict(
            color='lightsalmon',  # Lighter color for actual pre-forecast scatter
            size=2
        )
    )

    # Update the figure with these lines
    fig.add_traces([forecast_line, actual_test_line, actual_forecast_line])

    # Render the figure
    st.plotly_chart(fig)

    # Display a summary of settings
    st.subheader("Model Settings Summary")
    settings_columns = {
        "Weekly Seasonality": str(weekly_seasonality),
        "Yearly Seasonality": str(yearly_seasonality),
        "Monthly Seasonality": str(monthly_seasonality),
        "Algorithm Updates": str(alg_updates),
        "Referring Domains": str(referring_domains_check),
        "Outlier Adjustment": str(outlier_adjustment),
        "Seasonality Mode": seasonality_mode,
        "Changepoint Range": changepoint_range,
        "Changepoint Scale": changepoint_prior_scale,
        "Seasonality Prior Scale": seasonality_prior_scale,
        "Cap": cap if use_logistic_growth else "Not used",
        "Floor": floor if use_logistic_growth else "Not used"
    }
    # Ensure all values are converted to strings for Arrow compatibility
    settings_df = pd.DataFrame(settings_columns.items(), columns=["Parameter", "Value"]).astype(str)
    st.table(settings_df)

else:
    st.info("Please upload a valid CSV file.")