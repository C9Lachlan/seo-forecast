import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to load and prepare data
def load_and_prepare_data(traffic_file, keywords_file):
    # Load traffic data, skipping the 2nd and 3rd rows, but keeping the header
    traffic_data = pd.read_csv(traffic_file, skiprows=[1, 2])

    # Rename 'Metric' to 'Date' and check the columns again
    traffic_data.rename(columns={'Metric': 'Date'}, inplace=True)

    # Convert 'Date' to datetime format
    traffic_data['Date'] = pd.to_datetime(traffic_data['Date'])

    # Ensure the data is sorted by date
    traffic_data = traffic_data.sort_values('Date')

    # Rename columns for Prophet
    traffic_data = traffic_data.rename(columns={
        'Date': 'ds', 
        ' Avg. organic traffic': 'y', 
        ' Referring domains': 'referring_domains'
    })

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
    
    # Fit the model
    model.fit(traffic_data)
    
    return model

# Function for future predictions
def make_future_predictions(model, traffic_data, referring_domains, periods=365):
    future = model.make_future_dataframe(periods=periods)

    # Use the last known referring domains value for the future predictions
    last_referring_domains_value = referring_domains[-1] if len(referring_domains) > 0 else 0
    future['referring_domains'] = last_referring_domains_value
    
    # Generate predictions
    forecast = model.predict(future)
    
    # Ensure predictions do not go below 0
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    return forecast

# Function to calculate total traffic and differences
def calculate_total_traffic_and_difference(forecast_with_campaign, forecast_without_campaign):
    # Merge the two DataFrames on date
    merged_forecast = pd.merge(
        forecast_with_campaign[['ds', 'yhat']], 
        forecast_without_campaign[['ds', 'yhat']], 
        on='ds', 
        suffixes=('_with', '_without')
    )

    # Calculate total traffic for both scenarios
    total_traffic_with = merged_forecast['yhat_with'].sum()
    total_traffic_without = merged_forecast['yhat_without'].sum()

    # Calculate the differences in traffic, ensuring we avoid negative values
    merged_forecast['difference'] = merged_forecast['yhat_with'] - merged_forecast['yhat_without']
    # Only keep positive differences to reflect the incremental impact
    total_difference = merged_forecast['difference'][merged_forecast['difference'] > 0].sum()
    
    return total_traffic_with, total_traffic_without, total_difference

# Main Streamlit app
def main():
    st.title("Organic Traffic Prediction")
    
    # File upload for traffic data
    traffic_file = st.file_uploader("Upload Traffic CSV", type="csv")
    
    # File upload for keywords data
    keywords_file = st.file_uploader("Upload Keywords CSV", type="csv")
    
    # User inputs for marketing campaign date
    campaign_date = st.date_input("Enter the Campaign Start Date")

    if traffic_file and keywords_file:
        traffic_data, market_cap = load_and_prepare_data(traffic_file, keywords_file)
        
        # Check if 'Referring domains' exists in the DataFrame
        if 'referring_domains' not in traffic_data.columns:
            st.error("The column 'Referring domains' is not found in the traffic data. Please check the CSV format.")
            return
        
        # Extract referring domains
        referring_domains = traffic_data['referring_domains'].values
        
        # Create and fit the model to predict with the campaign
        model_with_campaign = create_prophet_model(traffic_data)
        forecast_with_campaign = make_future_predictions(model_with_campaign, traffic_data, referring_domains)
        
        # Create a subset of the traffic data with only pre-campaign data
        pre_campaign_data = traffic_data[traffic_data['ds'] < pd.to_datetime(campaign_date)]
        
        # Create and fit the model to predict without the campaign
        model_without_campaign = create_prophet_model(pre_campaign_data)
        forecast_without_campaign = make_future_predictions(model_without_campaign, pre_campaign_data, referring_domains)
        
        # Calculate total traffic and differences
        total_traffic_with, total_traffic_without, total_difference = calculate_total_traffic_and_difference(forecast_with_campaign, forecast_without_campaign)

        # Display total traffic and difference results
        st.write(f"Total Traffic with Campaign: {total_traffic_with:.2f}")
        st.write(f"Total Traffic without Campaign: {total_traffic_without:.2f}")
        st.write(f"Total Incremental Difference in Predicted Traffic Due to Campaign: {total_difference:.2f}")

        # Plot the predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        # Prophets plot method does not support setting colors directly so we will plot directly
        ax.plot(forecast_with_campaign['ds'], forecast_with_campaign['yhat'], label='Predictions with Campaign', color='blue')
        ax.plot(forecast_without_campaign['ds'], forecast_without_campaign['yhat'], label='Predictions without Campaign', color='orange')
        
        # Customize the plot
        plt.axvline(pd.to_datetime(campaign_date), color='red', linestyle='--', label='Campaign Start Date')
        plt.title("Predicted Traffic with and without Campaign")
        plt.ylim(bottom=0)  # Ensure the y-axis does not go below 0
        plt.legend()
        
        st.pyplot(fig)

        # Show Market Cap
        st.write("Total Market Cap from Keywords: ", market_cap)

if __name__ == "__main__":
    main()