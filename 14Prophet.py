import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

# Streamlit setup
st.title("Moses")

# Sidebar inputs
st.sidebar.header("User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload 'Ahrefs Traffic Data' CSV file", type=["csv"])

# Additional inputs for splitting and forecasting
test_data_end_date = st.sidebar.date_input("Test Data End Date")  # Change days to a calendar date
forecast_range_days = int(st.sidebar.number_input("Forecast Range (days)", min_value=1, value=182))

# Optional parameters
weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=False)
yearly_seasonality = st.sidebar.checkbox("Yearly Seasonality", value=True)
monthly_seasonality = st.sidebar.checkbox("Monthly Seasonality", value=False)
alg_updates = st.sidebar.checkbox("Algorithm Updates")
referring_domains_check = st.sidebar.checkbox("Referring Domains", value=False)
outlier_adjustment = st.sidebar.checkbox("Outlier Detection and Adjustment")

seasonality_mode = st.sidebar.selectbox("Seasonality Mode", options=["additive", "multiplicative"])
changepoint_range = float(st.sidebar.text_input("Changepoint Allowed Range", "0.9"))
changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Trend Flexibility)", min_value=0.01, max_value=10.0, value=0.05, step=0.01)
seasonality_prior_scale = st.sidebar.slider("Seasonality Prior Scale (Seasonal Flexability)", min_value=0.01, max_value=10.0, value=10.0, step=0.1)

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
    
    # Sort the data
    df.sort_values('ds', inplace=True)

    # Split data into forecast_data and test_data
    test_data_end_date = pd.to_datetime(test_data_end_date)  # Convert to datetime
    if test_data_end_date in df['ds'].values:  # Check if the date is in the data
        test_data = df[df['ds'] > test_data_end_date]
        forecast_data = df[df['ds'] <= test_data_end_date]
    else:
        forecast_data = df  # If not, use the entire dataset for training
        test_data = pd.DataFrame()

    # Fit the model
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
        m.add_country_holidays(country_name='AU')
        m = Prophet(holidays=holidays)
    if outlier_adjustment:
        pass

    if use_logistic_growth:
        forecast_data.loc[:, 'cap'] = cap
        forecast_data.loc[:, 'floor'] = floor

    m.fit(forecast_data)

    future = m.make_future_dataframe(periods=forecast_range_days)
    if referring_domains_check:
        future['referring_domains'] = forecast_data['referring_domains'].mean()
    
    if use_logistic_growth:
        future.loc[:, 'cap'] = cap
        future.loc[:, 'floor'] = floor
    
    forecast = m.predict(future)

    # Only calculate MAPE if test data is available
    if not test_data.empty:
        # Select only the forecast values that correspond to the `test_data` period
        forecast_for_test_period = forecast[forecast['ds'].isin(test_data['ds'])]

        y_true = test_data['y'].values
        y_pred = forecast_for_test_period['yhat'].values

        test_data = test_data.copy()
        test_data['MAPE'] = abs((y_true - y_pred) / y_true) * 100

        weighted_mape = round((test_data['y'] * test_data['MAPE']).sum() / test_data['y'].sum(), 2)
        total_difference = round((y_true - y_pred).sum(), 2)
        percentage_difference = round((total_difference / test_data['y'].sum()) * 100, 2)
        
        st.subheader("WAPE (Weighted Average Percentage Error)")
        st.text(f"{weighted_mape} %")

        st.subheader("Total Difference between Forecast and Actual")
        st.text(f"{total_difference} page views ({percentage_difference}%)")
    
    # Plot the results
    st.subheader("Organic Traffic Forecast")
    fig = plot_plotly(m, forecast)

    # Adjust the layout to start Y-axis at 0 and define a brighter color for the forecast
    fig.update_layout(
        yaxis=dict(range=[0, max(forecast['yhat'].max(), test_data['y'].max() if not test_data.empty else forecast['yhat'].max())]),
        template="plotly"
    )

    # Customize the forecast line and actual data points color
    forecast_line = go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='skyblue', width=2)
    )

    actual_forecast_line = go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['y'],
        mode='markers',
        name='Actual Forecast',
        marker=dict(
            color='white',
            size=2
        )
    )

    if not test_data.empty:
        actual_test_line = go.Scatter(
            x=test_data['ds'],
            y=test_data['y'],
            mode='markers',
            name='Actual Test',
            marker=dict(
                color='salmon',
                size=2
            )
        )
        fig.add_traces([forecast_line, actual_forecast_line, actual_test_line])
    else:
        fig.add_traces([forecast_line, actual_forecast_line])

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
        "Floor": floor if use_logistic_growth else "Not used",
        "Test Data End Date": test_data_end_date,
        "Forecast Range Days": forecast_range_days
    }
    settings_df = pd.DataFrame(settings_columns.items(), columns=["Parameter", "Value"]).astype(str)
    st.table(settings_df)

else:
    st.info("Please upload a valid CSV file.")