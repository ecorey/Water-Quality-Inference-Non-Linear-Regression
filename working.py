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
    Train models and make predictions with separate training and testing loops
    """
    from sklearn.linear_model import LinearRegression
    
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    
    # Prepare results
    predictions = []
    
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
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            # Training loop
            epochs = 50
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
                
                # Print progress every 10 epochs
                if epoch % 10 == 0 and station == stations[0]:
                    print(f"Station {station}, Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
            
            # Testing/Prediction
            model.eval()
            
            # Get latest data for prediction (from the most recent year)
            latest_data = station_data[station_data['Year'] == train_up_to_year].iloc[0][feature_cols].values.reshape(1, -1)
            
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
                
        except Exception as e:
            print(f"Error processing station {station}: {e}")
    
    # Create result DataFrame
    results_df = pd.DataFrame(predictions)
    
    # Save to CSV
    output_file = f'p90_predictions_{predict_for_year}_using_data_through_{train_up_to_year}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Saved {len(results_df)} predictions to {output_file}")
    
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
    
    print(f"\n=== Predicting for 2021 using data through 2020 ===")
    
    # Process data
    train_data, stations, station_coords = process_data(all_data, 
                                                        train_up_to_year=2020, 
                                                        predict_for_year=2021)
    

    # Train models and make predictions
    results = train_and_predict(train_data, 
                                stations, 
                                station_coords, 
                                train_up_to_year=2020, 
                                predict_for_year=2021)
    

    # Report overall time
    elapsed_time = time.time() - start_time
    print(f"\nAll processing completed in {elapsed_time:.2f} seconds")