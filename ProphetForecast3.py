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
    traffic_data = traffic_data.rename(columns={'Date': 'ds', ' Avg. organic traffic': 'y', ' Referring domains': 'referring_domains'})
    
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
    
    # Use the last known referring domains value for the future predictions
    last_referring_domains_value = referring_domains[-1] if len(referring_domains) > 0 else 0
    
    # Assign the referring domains and initialize algorithm_change
    future['referring_domains'] = last_referring_domains_value
    future['algorithm_change'] = 0
    
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

# Function to compute effect size
def calculate_effect_size(forecast, campaign_date):
    before_campaign = forecast[forecast['ds'] < campaign_date]
    after_campaign = forecast[forecast['ds'] >= campaign_date]

    if not before_campaign.empty and not after_campaign.empty:
        # Compute average traffic before and after campaign
        avg_before = before_campaign['yhat'].mean()
        avg_after = after_campaign['yhat'].mean()
        effect_size = avg_after - avg_before
        return avg_before, avg_after, effect_size
    else:
        return None, None, None  # Handle case where there's no data

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
        
        # Create and fit the model
        model, algorithm_change_dates = create_prophet_model(traffic_data)
        
        # Make future predictions
        forecast = make_future_predictions(model, traffic_data, referring_domains, algorithm_change_dates)
        
        # Calculate effect size
        avg_before, avg_after, effect_size = calculate_effect_size(forecast, pd.to_datetime(campaign_date))

        # Display effect size results
        if avg_before is not None and avg_after is not None:
            st.write("Average Traffic Before Campaign: {:.2f}".format(avg_before))
            st.write("Average Traffic After Campaign: {:.2f}".format(avg_after))
            st.write("Effect Size of the Campaign: {:.2f}".format(effect_size))
        else:
            st.error("No sufficient data to calculate effect size.")

        # Plot the predictions
        fig = model.plot(forecast)
        plt.ylim(bottom=0)  # ensure the y-axis does not go below 0
        st.pyplot(fig)
        
        # Show Market Cap
        st.write("Total Market Cap from Keywords: ", market_cap)

if __name__ == "__main__":
    main()