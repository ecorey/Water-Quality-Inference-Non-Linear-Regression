"""
This script is used to test the sequential model for predicting P90 scores using train_test_split.
"""

import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from tqdm.auto import tqdm
from torchmetrics import MeanSquaredError

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

###################
#### FUNCTIONS ####
###################

def process_data(data, test_size=0.2, random_state=42):
    """
    Process data and prepare for training and testing using train_test_split

    Args:
        data: Input data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of processed data and station coordinates
    """
    
    # drop unnecessary columns keeping geographical data
    data_columns = data.columns.tolist()
    columns_to_drop = [col for col in drop_columns if col in data_columns]
    data = data.drop(columns=columns_to_drop)
    
    # Replace inf values with NaN
    data = data.replace([np.inf, -np.inf], np.nan)
    
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
    stations = data['Station'].unique()
    print(f"Processing {len(stations)} stations")
    
    return data, stations, station_coords


def train_and_evaluate(data, stations, station_coords, test_size=0.2, random_state=42,
                       feature_cols=['GM', 'SDV', 'MAX_', 'Count_', 'MFCount', 'Year']):
    """
    Train models and evaluate them using train_test_split for each station.
    
    Args:
        data: Processed data
        stations: List of stations
        station_coords: Dictionary of station coordinates
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        feature_cols: List of feature columns to use for training

    Returns:
        DataFrame with predictions and evaluation metrics
    """
    
    # Set the random seed for reproducibility
    torch.manual_seed(random_state)
    
    # Prepare results
    predictions = []
    limited_data_stations = []
    error_stations = []
    
    # Progress tracking
    station_iterator = tqdm(stations, desc="Processing stations", total=len(stations))

    for station in station_iterator:
        try:
            # Get data for this station
            station_data = data[data['Station'] == station].sort_values('Year')
            
            # Skip if not enough data points
            if len(station_data) < 5:  # Need at least some data for meaningful split
                limited_data_stations.append(f"{station}: only {len(station_data)} records")
                continue
            
            # Prepare input data
            X = station_data[feature_cols].values
            y = station_data['P90'].values
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert to tensors
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
            
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
                y_pred = model(X_train_tensor)
                
                # Calculate loss
                loss = criterion(y_pred, y_train_tensor)
                
                # Calculate training accuracy
                mse_metric = MeanSquaredError().to(device)
                mse_value = mse_metric(y_pred, y_train_tensor)

                max_expected_mse = 100.0  
                accuracy_train = 100 * (1 - min(mse_value / max_expected_mse, 1.0))
                
                # Optimizer zero grad
                optimizer.zero_grad()
                
                # Backpropagation
                loss.backward()
                
                # Optimizer step
                optimizer.step()
                
                # Print progress every 10 epochs for first station only
                if epoch % 10 == 0 and station == stations[0]:
                    print(f"Station {station}, Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Train Accuracy: {accuracy_train:.2f}%")
            
            # Final evaluation
            model.eval()
            with torch.inference_mode():
                # Get predictions on test set
                test_preds = model(X_test_tensor)
                
                # Calculate test accuracy
                test_mse = mse_metric(test_preds, y_test_tensor)
                accuracy_test = 100 * (1 - min(test_mse / max_expected_mse, 1.0))
                
                # Store predictions and actual values
                for i in range(len(X_test)):
                    year = int(X_test[i][feature_cols.index('Year')] if 'Year' in feature_cols else 0)
                    prediction = test_preds[i].item()
                    prediction = max(0, prediction)
                    prediction = round(prediction, 1)
                    
                    # Create result entry
                    result_entry = {
                        'Station': station,
                        'Year': year,
                        'Actual_P90': y_test[i],
                        'Predicted_P90': prediction,
                        'Model_Accuracy': float(accuracy_test.cpu().numpy()) if isinstance(accuracy_test, torch.Tensor) else accuracy_test
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
    
    # Calculate RMSE and other metrics
    if 'Actual_P90' in results_df.columns and 'Predicted_P90' in results_df.columns:
        results_df['Error'] = results_df['Predicted_P90'] - results_df['Actual_P90']
        results_df['AbsError'] = results_df['Error'].abs()
        results_df['SquaredError'] = results_df['Error'] ** 2
        
        rmse = np.sqrt(results_df['SquaredError'].mean())
        mae = results_df['AbsError'].mean()
        
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    # Print summary statistics
    print("\n===== SUMMARY =====")
    print(f"Total stations processed: {len(stations)}")
    print(f"Successful predictions with neural network: {len(stations) - len(limited_data_stations) - len(error_stations)}")
    print(f"Stations with limited data: {len(limited_data_stations)}")
    print(f"Stations with errors: {len(error_stations)}")
    
    # Print examples of stations with issues
    if limited_data_stations:
        print(f"\nExample stations with limited data (showing {min(5, len(limited_data_stations))} of {len(limited_data_stations)}):")
        for station_info in limited_data_stations[:5]:
            print(f"  {station_info}")
            
    if error_stations:
        print(f"\nExample stations with errors (showing {min(5, len(error_stations))} of {len(error_stations)}):")
        for station_info in error_stations[:5]:
            print(f"  {station_info}")
    
    if overall_accuracy is not None:
        print(f"\nOverall model accuracy (across all stations): {overall_accuracy:.2f}%")
    
    # Save to CSV
    output_file = f'p90_predictions_using_train_test_split.csv'
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
    
    print(f"\n=== Predicting P90 values using train_test_split ===")
    
    # process data
    data, stations, station_coords = process_data(all_data)
    
    # train models and evaluate
    results = train_and_evaluate(
        data, 
        stations, 
        station_coords, 
        test_size=0.2,  # 20% of data used for testing
        random_state=42
    )
    
    # overall time
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")