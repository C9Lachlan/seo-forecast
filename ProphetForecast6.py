import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Organic Traffic Prediction with Prophet")

st.sidebar.title("Configuration")

# CSV input in sidebar
uploaded_file = st.sidebar.file_uploader("Upload 'Ahrefs Traffic Data' CSV", type="csv")

# Checkboxes
weekly_seasonality = st.sidebar.checkbox("Weekly seasonality", value=False)

algorithm_updates = st.sidebar.checkbox("Algorithm updates", value=False)

referring_domains_checkbox = st.sidebar.checkbox("Referring domains", value=False)

# Text inputs for changepoint range and trend flexibility
changepoint_range = st.sidebar.number_input("Changepoint Range", min_value=0.0, max_value=1.0, value=0.9)
trend_flexibility = st.sidebar.number_input("Trend flexibility", min_value=0.0, value=0.05)

if uploaded_file is not None:
    # Read the CSV, skipping second and third lines
    df = pd.read_csv(uploaded_file, skiprows=[1, 2])

    # Use only 'Metric', 'Referring domains', and 'Avg. organic traffic' columns
    df = df[['Metric', ' Referring domains', ' Organic traffic']]

    # Rename columns
    df = df.rename(columns={'Metric': 'ds', ' Organic traffic': 'y'})

    # Convert 'ds' to datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Drop rows with NaN in 'ds' or 'y'
    df = df.dropna(subset=['ds', 'y'])

    # Sort by date
    df = df.sort_values('ds')

    # **Display the DataFrame**
    st.write("### Data Preview:")
    st.dataframe(df)

    # Display Data Summary
    st.write("### Data Summary:")
    st.write(df.describe())
    st.write(f"Total data points: {len(df)}")

    # Handle algorithm updates
    if algorithm_updates:
        # List of algorithm update dates
        algorithm_update_dates = [
            '2024-11-11',
            '2024-08-15',
            '2024-06-20',
            '2024-05-14',
            '2024-05-06',
            '2024-03-05',
            '2023-11-08',
            '2023-11-02',
            '2023-10-05',
            '2023-10-04',
            '2023-09-14',
            '2023-08-22',
            '2023-04-12',
            '2023-03-15',
            '2023-02-21',
            '2022-12-14',
            '2022-12-05',
            '2022-10-19',
            '2022-09-20',
            '2022-09-12'
        ]

        # Convert to datetime
        algorithm_update_dates = pd.to_datetime(algorithm_update_dates)

        # Create 'algorithm_update' column
        df['algorithm_update'] = df['ds'].isin(algorithm_update_dates).astype(int)
    else:
        # Remove 'algorithm_update' column if it exists
        if 'algorithm_update' in df.columns:
            df = df.drop('algorithm_update', axis=1)

    if not referring_domains_checkbox:
        # Remove 'Referring domains' column
        if 'Referring domains' in df.columns:
            df = df.drop('Referring domains', axis=1)

    # **Display the DataFrame after processing**
    st.write("### Processed Data:")
    st.dataframe(df)

    # Determine the length of the dataset
    data_length = len(df)

    # Ensure we have enough data points
    if data_length < 3:
        st.error("Not enough data to build the model. Please upload a CSV file with at least 3 data points.")
    else:
        # Adjust test_length based on data size
        test_length = min(182, data_length // 2)

        # Split data into forecast_data and test_data
        forecast_data = df[:-test_length].copy()
        test_data = df[-test_length:].copy()

        # **Display the Forecast and Test DataFrames**
        st.write("### Forecast Data:")
        st.dataframe(forecast_data)

        st.write("### Test Data:")
        st.dataframe(test_data)

        # Initialize Prophet model with specified parameters
        m = Prophet(
            weekly_seasonality=weekly_seasonality,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=trend_flexibility,
            seasonality_mode='additive'
        )

        if algorithm_updates:
            m.add_regressor('algorithm_update')

        if referring_domains_checkbox:
            m.add_regressor(' Referring domains')

        # **Fit the model inside a try-except block to catch errors**
        try:
            with st.spinner('Training the model...'):
                m.fit(forecast_data)
        except ValueError as e:
            st.error(f"Error fitting the model: {e}")
            st.stop()

        # Prepare future dataframe
        future = test_data[['ds']].copy()

        if algorithm_updates:
            future['algorithm_update'] = test_data['algorithm_update']
        if referring_domains_checkbox:
            future[' Referring domains'] = test_data[' Referring domains']

        # Make the prediction
        forecast = m.predict(future)

        # Merge forecast with actuals
        forecast['y'] = test_data['y'].values

        # Compute MAPE
        forecast['MAPE'] = np.where(forecast['y'] == 0, np.nan,
                                    abs((forecast['y'] - forecast['yhat']) / forecast['y']) * 100)

        # Compute WAPE
        WAPE = np.sum(np.abs(forecast['y'] - forecast['yhat'])) / np.sum(forecast['y'])
        WAPE_percentage = WAPE * 100

        # Plotting
        # Combine forecast_data and test_data for plotting
        full_data = pd.concat([forecast_data, test_data])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(full_data['ds'], full_data['y'], label='Historical Data')
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        ax.plot(forecast['ds'], forecast['y'], label='Actual')
        ax.set_xlabel('Date')
        ax.set_ylabel('Traffic')
        ax.legend()

        st.pyplot(fig)

        # Display MAPE table
        st.write("### Mean Absolute Percentage Error (MAPE) for each day:")
        st.dataframe(forecast[['ds', 'y', 'yhat', 'MAPE']])

        # Display WAPE
        st.write(f"### Weighted Average Percentage Error (WAPE): {WAPE_percentage:.2f}%")