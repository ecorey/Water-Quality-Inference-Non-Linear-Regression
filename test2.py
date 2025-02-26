import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time



###################
#### FUNCTIONS ####
###################

# function to predict for a specific test year using data up to a cutoff
def predict_custom_year_range(data, train_up_to_year, predict_for_year):
    """
    Helper function to make predictions for a specific test year
    using data only up to a certain training cutoff year
    """
    print(f"\n=== Predicting for {predict_for_year} using data through {train_up_to_year} ===")
    
    # Create a lookup dictionary for coordinates
    station_coords = {}
    for _, row in data.iterrows():
        station = row['Station']
        if 'Lat_DD' in data.columns and 'Long_DD' in data.columns:
            lat = row.get('Lat_DD')
            lon = row.get('Long_DD')
            if pd.notna(lat) and pd.notna(lon):
                station_coords[station] = (lat, lon)
    
    # Run the prediction
    results = process_all_stations_linear(
        data,
        train_up_to_year=train_up_to_year,
        predict_for_year=predict_for_year
    )
    
    # Add coordinate columns if available
    if station_coords:
        results['Lat_DD'] = results['Station'].apply(
            lambda x: station_coords.get(x, (None, None))[0] if x in station_coords else None)
        results['Long_DD'] = results['Station'].apply(
            lambda x: station_coords.get(x, (None, None))[1] if x in station_coords else None)
    
    # Save to CSV
    output_file = f'p90_predictions_{predict_for_year}_using_data_through_{train_up_to_year}.csv'
    results.to_csv(output_file, index=False)
    print(f"Saved {len(results)} predictions to {output_file}")
    
    return results



def process_all_stations_linear(data, feature_cols=['GM', 'SDV', 'MAX_', 'Count_', 'MFCount', 'Year'], 
                               train_up_to_year=None, predict_for_year=None):
    """
    A simpler alternative using linear regression instead of neural networks
    
    Parameters:
    -----------
    data : DataFrame
        The dataset with all stations
    feature_cols : list
        List of feature column names to use for prediction
    train_up_to_year : int or None
        If specified, only use data up to this year for training
    predict_for_year : int or None
        If specified, predict for this specific year (should be train_up_to_year + 1)
    """
    from sklearn.linear_model import LinearRegression
    
    # Create a copy of the data with the geo columns preserved
    geo_data = data.copy()
    
    # Drop unnecessary columns, but keep geographical data
    data_columns = data.columns.tolist()
    columns_to_drop = [col for col in drop_columns if col in data_columns]
    data = data.drop(columns=columns_to_drop)
    
    # Replace inf values with NaN
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Determine the prediction year
    if train_up_to_year is None:
        # If not specified, use the latest year in the data
        train_up_to_year = data['Year'].max()
    
    if predict_for_year is None:
        # If not specified, predict for the next year after training
        predict_for_year = train_up_to_year + 1
    
    print(f"Training with data up to {train_up_to_year}, predicting for {predict_for_year}")
    
    # Filter data for training (only use data up to train_up_to_year)
    train_data = data[data['Year'] <= train_up_to_year]
    
    # Get unique stations
    stations = train_data['Station'].unique()
    print(f"Processing {len(stations)} stations using linear regression")
    
    # Prepare results
    predictions = []
    
    # Create a lookup for geographical information
    geo_lookup = {}
    for station in stations:
        station_geo = geo_data[geo_data['Station'] == station]
        if len(station_geo) > 0:
            lat = station_geo['Lat_DD'].iloc[0] if 'Lat_DD' in geo_data.columns else None
            lon = station_geo['Long_DD'].iloc[0] if 'Long_DD' in geo_data.columns else None
            geo_lookup[station] = {'Lat_DD': lat, 'Long_DD': lon}
    
    for station in stations:
        # Get data for this station
        station_data = train_data[train_data['Station'] == station].sort_values('Year')
        
        # Need at least 2 years of data
        if len(station_data) < 2:
            continue
        
        try:
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
            
            # Skip if we don't have the most recent year for prediction basis
            if station_data['Year'].max() < train_up_to_year:
                print(f"Skipping station {station} - missing data for {train_up_to_year}")
                continue
                
            # Prepare data
            X = station_data[feature_cols].values
            y = station_data['P90'].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train linear model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Get latest data for prediction (from the most recent year)
            latest_data = station_data[station_data['Year'] == train_up_to_year].iloc[0][feature_cols].values.reshape(1, -1)
            
            # Modify the Year value to be the prediction year
            latest_data_copy = latest_data.copy()
            year_index = feature_cols.index('Year') if 'Year' in feature_cols else None
            if year_index is not None:
                latest_data_copy[0, year_index] = predict_for_year
                
            latest_scaled = scaler.transform(latest_data_copy)
            
            # Predict for the prediction year
            prediction = model.predict(latest_scaled)[0]
            prediction = max(0, prediction)  # Ensure non-negative
            prediction = round(prediction, 1)
            
            # Get geographical info for this station
            geo_info = geo_lookup.get(station, {})
            
            # Add to results
            result_entry = {
                'Station': station,
                'Year': predict_for_year,
                'Predicted_P90': prediction
            }
            
            # Add geographical info if available
            for geo_col in ['Lat_DD', 'Long_DD']:
                if geo_col in geo_info:
                    result_entry[geo_col] = geo_info[geo_col]
            
            predictions.append(result_entry)
            
            # Save intermediate results
            if len(predictions) % 50 == 0:
                temp_df = pd.DataFrame(predictions)
                temp_df.to_csv('linear_p90_predictions_temp.csv', index=False)
                
        except Exception as e:
            print(f"Error processing station {station} with linear model: {e}")
    
    # Create final DataFrame
    results_df = pd.DataFrame(predictions)
    return results_df



# function to train a model for a specific station
def train_station_model(station_data, feature_cols, target_col='P90', epochs=20):
    # Prepare data
    X = station_data[feature_cols].values
    y = station_data[target_col].values.reshape(-1, 1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Create and train model
    model = StationP90Model(len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model, scaler



# function for prediction 
def predict_next_year(model, scaler, latest_data, feature_cols):
    # Scale the input features
    X = latest_data[feature_cols].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Convert to tensor and predict
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(X_tensor).item()
    
    return max(0, prediction)  # Ensure non-negative P90



# function to process all stations with batching
def process_all_stations(data, feature_cols=['GM', 'SDV', 'MAX_', 'Count_', 'MFCount', 'Year'], 
                          batch_size=100):
    # Drop unnecessary columns
    data = data.drop(columns=[col for col in drop_columns if col in data.columns])
    
    # Get unique stations
    stations = data['Station'].unique()
    print(f"Processing {len(stations)} stations in batches of {batch_size}")
    
    # Prepare results
    predictions = []
    
    # Process in batches
    for i in range(0, len(stations), batch_size):
        batch_stations = stations[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(stations)-1)//batch_size + 1}")
        
        for station in batch_stations:
            # Get data for this station
            station_data = data[data['Station'] == station].sort_values('Year')
            
            # Need at least 2 years of data
            if len(station_data) < 2:
                continue
            
            try:
                # Get last year's data
                last_year = station_data['Year'].max()
                next_year = last_year + 1
                
                # Train model on all available data
                model, scaler = train_station_model(station_data, feature_cols)
                
                # Get latest data point for prediction
                latest_data = station_data.iloc[-1]
                
                # Predict next year
                p90_prediction = predict_next_year(model, scaler, latest_data, feature_cols)
                
                # Add to results
                predictions.append({
                    'Station': station,
                    'Year': next_year,
                    'Predicted_P90': p90_prediction
                })
                
                # Save intermediate results after each batch
                if len(predictions) % 10 == 0:
                    temp_df = pd.DataFrame(predictions)
                    temp_df.to_csv('station_p90_predictions_temp.csv', index=False)
                
            except Exception as e:
                print(f"Error processing station {station}: {e}")
        
        # Save batch results
        batch_df = pd.DataFrame(predictions)
        batch_df.to_csv('station_p90_predictions_batch.csv', index=False)
    
    # Create final DataFrame
    results_df = pd.DataFrame(predictions)
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
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)
    


    
###################
###### MAIN #######
###################

# Main execution
if __name__ == "__main__":

    # Start timing
    start_time = time.time()
    

    #  prediction
    results = predict_custom_year_range(all_data, 2020, 2021)
    
    
    
    # report overall time
    elapsed_time = time.time() - start_time
    print(f"\nAll processing completed in {elapsed_time:.2f} seconds")