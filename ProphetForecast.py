import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to run the forecasting
def forecast_and_plot(df, metric):
    # Prepare the dataframe for Prophet
    df = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'})
    
    # Initialize and fit the Prophet model
    model = Prophet(weekly_seasonality=False)
    model.fit(df)
    
    # Create future dates for predictions
    future = model.make_future_dataframe(periods=300)  # Amount of forecasting
    forecast = model.predict(future)
    
    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f'{metric} Forecast')
    st.pyplot(fig)

# Streamlit interface
def main():
    st.title("Search Console Clicks and Impressions Forecasting")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Search Console CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Check if necessary columns exist
        if {'Date', 'Clicks', 'Impressions'}.issubset(df.columns):
            st.write("Data Preview:")
            st.write(df.head())
            
            # Convert 'Date' column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Plot forecast for Clicks
            st.subheader("Clicks Forecast")
            forecast_and_plot(df, 'Clicks')

            # Plot forecast for Impressions
            st.subheader("Impressions Forecast")
            forecast_and_plot(df, 'Impressions')
        else:
            st.error("The uploaded file must contain 'Date', 'Clicks', and 'Impressions' columns.")
            
if __name__ == "__main__":
    main()