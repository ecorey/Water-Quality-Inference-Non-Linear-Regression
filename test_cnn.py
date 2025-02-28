import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import os

###################
#### FUNCTIONS ####
###################

def process_data(data, train_up_to_year, predict_for_year, feature_cols=['GM', 'SDV', 'MAX_', 'Count_', 'MFCount', 'Year']):
    """
    Process data and prepare for training and testing
    """
    # Create a copy of the data with the geo columns preserved
    geo_data = data.copy()
    
    # Drop unnecessary columns, but keep geographical data
    data_columns = data.columns.tolist()
    columns_to_drop = [col for col in drop_columns if col in data_columns]
    data = data.drop(columns=columns_to_drop)
    
    # Replace inf values with NaN
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Filter data for training and testing
    train_data = data[data['Year'] <= train_up_to_year]
    
    # Create a lookup for geographical information
    station_coords = {}
    for _, row in data.iterrows():
        station = row['Station']
        if 'Lat_DD' in data.columns and 'Long_DD' in data.columns:
            lat = row.get('Lat_DD')
            lon = row.get('Long_DD')
            if pd.notna(lat) and pd.notna(lon):
                station_coords[station] = (lat, lon)
    
    # Get unique stations
    stations = train_data['Station'].unique()
    print(f"Processing {len(stations)} stations")
    
    return train_data, stations, station_coords

def train_and_predict(train_data, stations, station_coords, train_up_to_year, predict_for_year, 
                      feature_cols=['GM', 'SDV', 'MAX_', 'Count_', 'MFCount', 'Year']):
    """
    Train models and make predictions with separate training and testing loops.
    Handles stations with limited data using fallback methods.
    """
    from sklearn.linear_model import LinearRegression
    import signal
    from contextlib import contextmanager
    
    # Define a timeout handler
    class TimeoutException(Exception): pass
    
    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    
    # Prepare results
    predictions = []
    limited_data_stations = []
    error_stations = []
    timeout_stations = []
    
    # Default prediction fallback for stations with insufficient data
    overall_mean_p90 = train_data['P90'].mean()
    
    # Progress tracking
    total_stations = len(stations)
    processed_count = 0
    last_progress_update = time.time()
    
    for station in stations:
        processed_count += 1
        
        # Print progress every 50 stations or 30 seconds
        current_time = time.time()
        if processed_count % 50 == 0 or current_time - last_progress_update > 30:
            print(f"Progress: {processed_count}/{total_stations} stations ({processed_count/total_stations*100:.1f}%)")
            last_progress_update = current_time
        
        try:
            # Use a timeout of 60 seconds per station
            with time_limit(60):
                # Get data for this station
                station_data = train_data[train_data['Station'] == station].sort_values('Year')
                
                # Handle stations with limited data (fewer than 2 years)
                if len(station_data) < 2:
                    limited_data_stations.append(f"{station} (only {len(station_data)} records)")
                    
                    # For stations with only one data point, use that data directly
                    if len(station_data) == 1:
                        # Create a prediction based on the single observation
                        recent_p90 = station_data['P90'].iloc[0]
                        prediction = round(max(0, recent_p90), 1)  # Ensure non-negative and round
                        
                        result_entry = {
                            'Station': station,
                            'Year': predict_for_year,
                            'Predicted_P90': prediction,
                            'Note': 'Limited data prediction'
                        }
                        
                        # Add geographical info if available
                        if station in station_coords:
                            result_entry['Lat_DD'] = station_coords[station][0]
                            result_entry['Long_DD'] = station_coords[station][1]
                        
                        predictions.append(result_entry)
                        continue
                    
                    # For stations with no data, use overall mean as fallback
                    elif len(station_data) == 0:
                        prediction = round(max(0, overall_mean_p90), 1)
                        
                        result_entry = {
                            'Station': station,
                            'Year': predict_for_year,
                            'Predicted_P90': prediction,
                            'Note': 'No data available, using overall mean'
                        }
                        
                        # Add geographical info if available
                        if station in station_coords:
                            result_entry['Lat_DD'] = station_coords[station][0]
                            result_entry['Long_DD'] = station_coords[station][1]
                        
                        predictions.append(result_entry)
                        continue
                
                # Check for NaN values and handle them
                if station_data[feature_cols].isna().any().any():
                    # Fill NaN values in feature columns
                    station_data_filled = station_data.copy()
                    for col in feature_cols:
                        if station_data[col].isna().any():
                            # If a column has NaN, use the mean of that column for that station
                            mean_val = station_data[col].mean()
                            if np.isnan(mean_val):  # If all values are NaN
                                # Use overall mean from the dataset
                                mean_val = train_data[col].mean()
                                if np.isnan(mean_val):  # If still NaN
                                    mean_val = 0  # Default to 0
                            station_data_filled[col] = station_data[col].fillna(mean_val)
                    station_data = station_data_filled
                
                # Check if we have data for the train_up_to_year
                latest_year_data = station_data[station_data['Year'] == train_up_to_year]
                if len(latest_year_data) == 0:
                    # If we don't have data for the exact year, use the most recent year's data
                    max_year = station_data['Year'].max()
                    latest_year_data = station_data[station_data['Year'] == max_year]
                
                # Prepare input data
                X = station_data[feature_cols].values
                y = station_data['P90'].values
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Convert to tensors
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
                
                # Create the model
                model = StationP90Model(len(feature_cols))
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

                
                # Training loop
                epochs = 60
                for epoch in range(epochs):
                    # Training
                    model.train()
                    
                    # Forward pass
                    y_pred = model(X_tensor)
                    
                    # Calculate loss
                    loss = criterion(y_pred, y_tensor)
                    
                    # Optimizer zero grad
                    optimizer.zero_grad()
                    
                    # Backpropagation
                    loss.backward()
                    
                    # Optimizer step
                    optimizer.step()
                    
                    # Print progress every 10 epochs for first station only
                    if epoch % 10 == 0 and station == stations[0]:
                        print(f"Station {station}, Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
                
                # Testing/Prediction
                model.eval()
                
                # Get latest data for prediction (from the most recent year available)
                latest_data = latest_year_data.iloc[0][feature_cols].values.reshape(1, -1)
                
                # Modify the Year value to be the prediction year
                latest_data_copy = latest_data.copy()
                year_index = feature_cols.index('Year') if 'Year' in feature_cols else None
                if year_index is not None:
                    latest_data_copy[0, year_index] = predict_for_year
                
                latest_scaled = scaler.transform(latest_data_copy)
                X_test = torch.tensor(latest_scaled, dtype=torch.float32)
                
                # Make prediction
                with torch.inference_mode():
                    prediction = model(X_test).item()
                    prediction = max(0, prediction)  # Ensure non-negative
                    prediction = round(prediction, 1)  # Round to 1 decimal place
                
                # Create result entry
                result_entry = {
                    'Station': station,
                    'Year': predict_for_year,
                    'Predicted_P90': prediction
                }
                
                # Add geographical info if available
                if station in station_coords:
                    result_entry['Lat_DD'] = station_coords[station][0]
                    result_entry['Long_DD'] = station_coords[station][1]
                
                predictions.append(result_entry)
                
                # Save intermediate results
                if len(predictions) % 50 == 0:
                    temp_df = pd.DataFrame(predictions)
                    temp_df.to_csv('p90_predictions_temp.csv', index=False)
                
        except TimeoutException:
            timeout_stations.append(station)
            print(f"Timeout processing station {station}")
            
            # Use fallback prediction for timeout stations
            if station_data is not None and len(station_data) > 0:
                recent_p90 = station_data['P90'].mean()
            else:
                recent_p90 = overall_mean_p90
                
            prediction = round(max(0, recent_p90), 1)
            
            result_entry = {
                'Station': station,
                'Year': predict_for_year,
                'Predicted_P90': prediction,
                'Note': 'Timeout, using fallback value'
            }
            
            # Add geographical info if available
            if station in station_coords:
                result_entry['Lat_DD'] = station_coords[station][0]
                result_entry['Long_DD'] = station_coords[station][1]
            
            predictions.append(result_entry)
            
        except Exception as e:
            error_stations.append(f"{station}: {str(e)}")
            print(f"Error processing station {station}: {e}")
            
            # Use fallback prediction for error stations
            try:
                if 'station_data' in locals() and station_data is not None and len(station_data) > 0:
                    recent_p90 = station_data['P90'].mean()
                else:
                    recent_p90 = overall_mean_p90
                    
                prediction = round(max(0, recent_p90), 1)
                
                result_entry = {
                    'Station': station,
                    'Year': predict_for_year,
                    'Predicted_P90': prediction,
                    'Note': 'Error, using fallback value'
                }
                
                # Add geographical info if available
                if station in station_coords:
                    result_entry['Lat_DD'] = station_coords[station][0]
                    result_entry['Long_DD'] = station_coords[station][1]
                
                predictions.append(result_entry)
            except:
                print(f"Could not create fallback prediction for station {station}")
    
    # Create result DataFrame
    results_df = pd.DataFrame(predictions)
    
    # Print summary statistics
    print("\n===== SUMMARY =====")
    print(f"Total stations processed: {total_stations}")
    print(f"Successful predictions with neural network: {total_stations - len(limited_data_stations) - len(error_stations) - len(timeout_stations)}")
    print(f"Stations with limited data: {len(limited_data_stations)}")
    print(f"Stations with errors: {len(error_stations)}")
    print(f"Stations with timeouts: {len(timeout_stations)}")
    
    # Print examples of stations with issues
    if limited_data_stations:
        print(f"\nExample stations with limited data (showing {min(5, len(limited_data_stations))} of {len(limited_data_stations)}):")
        for station_info in limited_data_stations[:5]:
            print(f"  {station_info}")
            
    if error_stations:
        print(f"\nExample stations with errors (showing {min(5, len(error_stations))} of {len(error_stations)}):")
        for station_info in error_stations[:5]:
            print(f"  {station_info}")
    
    # Save to CSV
    output_file = f'p90_predictions_{predict_for_year}_using_data_through_{train_up_to_year}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(results_df)} predictions to {output_file}")
    
    return results_df

###################
####### DATA ######
###################

# File paths
file_list = [
    './data/2013_P90_Scores.csv', 
    './data/2014_P90_Scores.csv', 
    './data/2015_P90_Scores.csv', 
    './data/2016_P90_Scores.csv', 
    './data/2017_P90_Scores.csv', 
    './data/2018_P90_Scores.csv', 
    './data/2019_P90_Scores.csv', 
    './data/2020_P90_Scores.csv', 
    './data/2021_P90_Scores.csv', 
    './data/2022_P90_Scores.csv'
]

# columns to keep 
geo_columns = ['Lat_DD', 'Long_DD']

# columns to drop
drop_columns = ['X', 'Y', 'x', 'y', 'GlobalID', 'OBJECTID', 'Class', 'Appd_Std', 
                'Restr_Std', 'Min_Date', 'Grow_Area']

# load and combine all datasets
dataframes = []
for file in file_list:
    try:
        year = int(file.split('_')[0].split('/')[-1])
        df = pd.read_csv(file)
        
        # replace any infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # add year column
        df['Year'] = year
        dataframes.append(df)
    except Exception as e:
        print(f"Could not load {file}: {e}")

# combine all data
all_data = pd.concat(dataframes, ignore_index=True)

# fill NaN values for specific columns
numeric_cols = ['GM', 'SDV', 'MAX_', 'Count_', 'MFCount', 'P90']
for col in numeric_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(all_data[col].median())




###################
###### MODEL ######
###################

class StationP90Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        # For time series data, we need to determine the sequence length
        # Each data point will be a feature vector with 'input_size' dimensions
        # We'll reshape the input to treat it as a 1D sequence
        
        # The CNN expects input in the shape [batch_size, channels, sequence_length]
        # We'll use a dynamic approach to handle variable sequence lengths
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # The fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(32, 16),  # The input size will be determined dynamically
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        # x shape: [batch_size, features]
        
        # For Conv1d, we need [batch_size, channels, seq_length]
        # We'll transpose to make features the channels
        batch_size = x.size(0)
        
        # Reshape to [batch_size, features, 1]
        x = x.unsqueeze(2)
        
        # Transpose to [batch_size, features, 1]
        # This makes each feature a separate channel
        x = x.transpose(1, 2)
        
        # Pass through convolutional layers
        x = self.conv_layers(x)
        
        # Global average pooling to get fixed size regardless of input length
        x = torch.mean(x, dim=2)
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        
        return x 


###################
###### MAIN #######
###################

# Main execution
if __name__ == "__main__":
    # Start timing
    start_time = time.time()
    
    print(f"\n=== Predicting for 2021 using data through 2020 ===")
    
    # Process data
    train_data, stations, station_coords = process_data(all_data, 
                                                        train_up_to_year=2022, 
                                                        predict_for_year=2023)
    

    # Train models and make predictions
    results = train_and_predict(train_data, 
                                stations, 
                                station_coords, 
                                train_up_to_year=2022, 
                                predict_for_year=2023)
    

    # Report overall time
    elapsed_time = time.time() - start_time
    print(f"\nAll processing completed in {elapsed_time:.2f} seconds")