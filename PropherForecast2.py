import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to load and prepare data
def load_and_prepare_data(traffic_file, keywords_file):
    # Load traffic data, skipping the 2nd and 3rd rows, but keeping the header
    traffic_data = pd.read_csv(traffic_file, skiprows=[1, 2])
    
    # Print the columns to debug
    st.write("Traffic data columns:", traffic_data.columns)

    # Rename 'Metric' to 'Date' and check the columns again
    traffic_data.rename(columns={'Metric': 'Date'}, inplace=True)
    
    # Convert 'Date' to datetime format
    traffic_data['Date'] = pd.to_datetime(traffic_data['Date'])
    
    # Ensure the data is sorted by date
    traffic_data = traffic_data.sort_values('Date')
    
    # Rename columns for Prophet
    traffic_data = traffic_data.rename(columns={'Date': 'ds', ' Avg. organic traffic': 'y', ' Referring domains': 'referring_domains'})
    
    # Print the DataFrame for debugging
    st.write("Processed traffic data:", traffic_data.head())
    
    # Load keyword data
    keywords_data = pd.read_csv(keywords_file)
    
    # Calculate market cap
    market_cap = keywords_data['Volume'].sum()

    return traffic_data, market_cap

# Function to create and fit the model
def create_prophet_model(traffic_data):
    # Instantiate a Prophet model
    model = Prophet(seasonality_mode='additive', weekly_seasonality=False)
    
    # Add referring domains as an additional regressor
    model.add_regressor('referring_domains')
    
    # Add significant algorithm change dates as additional regressors
    algorithm_change_dates = [
        '2024-11-11', '2024-08-15', '2024-06-20', '2024-05-14', '2024-05-06',
        '2024-03-05', '2023-11-08', '2023-11-02', '2023-10-05', '2023-10-04',
        '2023-09-14', '2023-08-22', '2023-04-12', '2023-03-15', '2023-02-21',
        '2022-12-14', '2022-12-05', '2022-10-19', '2022-09-20', '2022-09-12'
    ]

    for date in algorithm_change_dates:
        traffic_data.loc[traffic_data['ds'] == date, 'algorithm_change'] = 1
    traffic_data['algorithm_change'].fillna(0, inplace=True)
    
    # Fit the model
    model.fit(traffic_data)
    
    return model, algorithm_change_dates

# Function for future predictions
def make_future_predictions(model, traffic_data, referring_domains, algorithm_change_dates, periods=365):
    future = model.make_future_dataframe(periods=periods)
    
    # Ensure to use the last known referring domains value for the future predictions
    last_referring_domains_value = referring_domains[-1] if len(referring_domains) > 0 else 0
    
    # Creating a full array of referring domains to match the length of future DataFrame
    future['referring_domains'] = last_referring_domains_value
    future['algorithm_change'] = 0  # Initialize algorithm_change as 0
    
    # Set algorithm change dates to 1 in the future DataFrame
    for date in algorithm_change_dates:
        if pd.to_datetime(date) in future['ds'].values:
            future.loc[future['ds'] == pd.to_datetime(date), 'algorithm_change'] = 1
            
    forecast = model.predict(future)

    # Ensure predictions do not go below 0
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    
    return forecast

# Main Streamlit app
def main():
    st.title("Organic Traffic Prediction")
    
    # File upload for traffic data
    traffic_file = st.file_uploader("Upload Traffic CSV", type="csv")
    
    # File upload for keywords data
    keywords_file = st.file_uploader("Upload Keywords CSV", type="csv")
    
    if traffic_file and keywords_file:
        traffic_data, market_cap = load_and_prepare_data(traffic_file, keywords_file)
        
        # Check if 'Referring domains' exists in the DataFrame
        if 'referring_domains' not in traffic_data.columns:
            st.error("The column 'Referring domains' is not found in the traffic data. Please check the CSV format.")
            return
        
        # Extract referring domains
        referring_domains = traffic_data['referring_domains'].values
        
        # Create and fit the model
        model, algorithm_change_dates = create_prophet_model(traffic_data)
        
        # Make future predictions
        forecast = make_future_predictions(model, traffic_data, referring_domains, algorithm_change_dates)
        
        # Plot the predictions
        fig = model.plot(forecast)
        st.pyplot(fig)
        
        # Show Market Cap
        st.write("Total Market Cap from Keywords: ", market_cap)

if __name__ == "__main__":
    main()