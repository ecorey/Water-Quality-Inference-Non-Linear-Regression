"""
This script is used to test the sequential model for predicting P90 scores for a given year using data through the given year.
"""

import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from tqdm.auto import tqdm
from torchmetrics import MeanSquaredError



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



###################
#### FUNCTIONS ####
###################

def process_data(data, train_up_to_year):
    """
    Process data and prepare for training and testing

    Args:
        data: Input data
        train_up_to_year: Year up to which to use data for training
    
    Returns:
        Tuple of train data, stations, and station coordinates

    """
    
    # drop unnecessary columns keeping geographical data
    data_columns = data.columns.tolist()
    columns_to_drop = [col for col in drop_columns if col in data_columns]
    data = data.drop(columns=columns_to_drop)
    
    # Replace inf values with NaN
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # filter data for training and testing
    train_data = data[data['Year'] <= train_up_to_year]
    
    # create a lookup for geographical information
    station_coords = {}
    for _, row in data.iterrows():
        station = row['Station']
        if 'Lat_DD' in data.columns and 'Long_DD' in data.columns:
            lat = row.get('Lat_DD')
            lon = row.get('Long_DD')
            if pd.notna(lat) and pd.notna(lon):
                station_coords[station] = (lat, lon)
    
    # get stations
    stations = train_data['Station'].unique()
    print(f"Processing {len(stations)} stations")
    
    return train_data, stations, station_coords


def train_and_predict(train_data, stations, station_coords, train_up_to_year, predict_for_year, 
                      feature_cols=['GM', 'SDV', 'MAX_', 'Count_', 'MFCount', 'Year']):
    """
    Train models and make predictions with separate training and testing loops.
    Handles stations with limited data using fallback methods.

    Args:
        train_data: Data for training
        stations: List of stations
        station_coords: Dictionary of station coordinates
        train_up_to_year: Year up to which to use data for training
        predict_for_year: Year for which to make predictions
        feature_cols: List of feature columns to use for training

    Returns:
        DataFrame with predictions
    """
    
   
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    
    # Prepare results
    predictions = []
    limited_data_stations = []
    error_stations = []
    timeout_stations = []
    skipped_stations = []

    total_stations = len(stations)

    
    
    # Progress tracking
    station_iterator = tqdm(stations, desc="Processing stations", total=len(stations))


    for station in station_iterator:

        try:

            # Get data for this station
            station_data = train_data[train_data['Station'] == station].sort_values('Year')
            

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
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
            
            # Create the model
            model = StationP90Model(len(feature_cols)).to(device)
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
                
                # Calculate accuracy
                mse_metric = MeanSquaredError().to(device)
                mse_value = mse_metric(y_pred, y_tensor)


                max_expected_mse = 100.0  # Set this based on your data's scale
                accuracy_calculated = 100 * (1 - min(mse_value / max_expected_mse, 1.0))

                
                # Optimizer zero grad
                optimizer.zero_grad()
                
                # Backpropagation
                loss.backward()
                
                # Optimizer step
                optimizer.step()
                
                # Print progress every 10 epochs for first station only
                if epoch % 10 == 0 and station == stations[0]:
                    print(f"Station {station}, Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy_calculated:.2f}%")
                
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
                X_test = torch.tensor(latest_scaled, dtype=torch.float32).to(device)
                


                # Make prediction
                with torch.inference_mode():
                    prediction = model(X_test).item()
                    
                    # Calculate model accuracy for this station
                    test_preds = model(X_tensor)
                    
                    # Calculate accuracy
                    mse_metric = MeanSquaredError().to(device)
                    test_mse = mse_metric(test_preds, y_tensor)
                    accuracy_calculated_test = 100 * (1 - min(test_mse / max_expected_mse, 1.0))
                    
                    prediction = max(0, prediction)  
                    prediction = round(prediction, 1)  
                
                if station == stations[0]:
                    print(f"Final test accuracy for {station}: {accuracy_calculated_test:.2f}%")
                
                # Create result entry
                result_entry = {
                    'Station': station,
                    'Year': predict_for_year,
                    'Predicted_P90': prediction,
                    'Model_Accuracy': float(accuracy_calculated_test.cpu().numpy()) if isinstance(accuracy_calculated_test, torch.Tensor) else accuracy_calculated_test
 
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
            error_stations.append(f"{station}: {str(e)}")
            print(f"Error processing station {station}: {e}")
            
    
    # Create result DataFrame
    results_df = pd.DataFrame(predictions)
    
    # Calculate overall accuracy
    valid_accuracies = results_df['Model_Accuracy'].dropna()
    overall_accuracy = valid_accuracies.mean() if len(valid_accuracies) > 0 else None
    
    # Print summary statistics
    print("\n===== SUMMARY =====")
    print(f"Total stations processed: {total_stations}")
    print(f"Successful predictions with neural network: {total_stations - len(limited_data_stations) - len(error_stations) - len(timeout_stations) - len(skipped_stations)}")
    print(f"Stations with limited data: {len(limited_data_stations)}")
    print(f"Stations with errors: {len(error_stations)}")
    print(f"Stations with timeouts: {len(timeout_stations)}")
    print(f"Stations skipped (no data): {len(skipped_stations)}")
    
    # Print examples of stations with issues
    if limited_data_stations:
        print(f"\nExample stations with limited data (showing {min(5, len(limited_data_stations))} of {len(limited_data_stations)}):")
        for station_info in limited_data_stations[:5]:
            print(f"  {station_info}")
            
    if error_stations:
        print(f"\nExample stations with errors (showing {min(5, len(error_stations))} of {len(error_stations)}):")
        for station_info in error_stations[:5]:
            print(f"  {station_info}")
            
    if skipped_stations:
        print(f"\nExample stations skipped (showing {min(5, len(skipped_stations))} of {len(skipped_stations)}):")
        for station_info in skipped_stations[:5]:
            print(f"  {station_info}")
    
    if overall_accuracy is not None:
        print(f"\nOverall model accuracy (across all stations): {overall_accuracy:.2f}%")


    
    # Save to CSV
    output_file = f'p90_predictions_{predict_for_year}_using_data_through_{train_up_to_year}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(results_df)} predictions to {output_file}")


    
    return results_df



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
####### DATA ######
###################

# file paths
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
###### MAIN #######
###################


if __name__ == "__main__":

    # start timing
    start_time = time.time()
    
    print(f"\n=== Predicting P90 values using data through given year ===")
    
    # process data
    train_data, stations, station_coords = process_data(all_data, 
                                                        train_up_to_year=2019)
    
    # train models and make predictions
    results = train_and_predict(train_data, 
                                stations, 
                                station_coords, 
                                train_up_to_year=2019, 
                                predict_for_year=2020)
    
    # overall time
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")