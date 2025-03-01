"""
This script is used to test the sequential model for predicting P90 and GM scores for a given year using data through the given year.
The model uses a non-linear regression neural network to infer the results.
"""

import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from tqdm.auto import tqdm
from torchmetrics import MeanSquaredError


# device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



###################
#### FUNCTIONS ####
###################

def process_data(train_up_to_year):
    """
    Processes the data and prepares it for training and testing in the neural network model.
    Args:
        train_up_to_year: Year up to which to use data for training
    
    Returns:
        Tuple of train data, stations, and station coordinates
    """
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

    # columns to drop
    drop_columns = ['X', 'Y', 'x', 'y', 'GlobalID', 'OBJECTID', 'Appd_Std', 
                    'Restr_Std', 'Min_Date', 'Class', 'Grow_Area']

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
    data = pd.concat(dataframes, ignore_index=True)

    # fill NaN values for specific columns
    numeric_cols = ['GM', 'SDV', 'MAX_', 'Count_', 'MFCount', 'P90']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
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
                      feature_cols_p90=['GM', 'SDV', 'MAX_', 'Count_', 'MFCount', 'Year'],
                      feature_cols_gm=['P90', 'SDV', 'MAX_', 'Count_', 'MFCount', 'Year']):
    """
    Trains models and makes predictions with separate training and testing loops.
    The model uses non-linear regression neural networks to predict both P90 and GM values.
    Handles stations that have limited data by using a simpler model.

    Args:
        train_data: Data for training
        stations: List of stations
        station_coords: Dictionary of station coordinates
        train_up_to_year: Year up to which to use data for training
        predict_for_year: Year for which to make predictions
        feature_cols_p90: List of feature columns to use for P90 prediction
        feature_cols_gm: List of feature columns to use for GM prediction

    Returns:
        DataFrame with predictions
    """
    
    # set the random seed for reproducibility
    torch.manual_seed(42)
    
    # list vars
    predictions = []
    limited_data_stations = []
    error_stations = []
    timeout_stations = []
    skipped_stations = []

    total_stations = len(stations)
    
    # tqdm progress bar
    station_iterator = tqdm(stations, desc="Processing stations", total=len(stations))

    for station in station_iterator:
        try:
            # get data for this station
            station_data = train_data[train_data['Station'] == station].sort_values('Year')
            
            # Skip if not enough data
            if len(station_data) == 0:
                skipped_stations.append(f"{station}: no data")
                continue
                
            # check if we have data for the train_up_to_year
            latest_year_data = station_data[station_data['Year'] == train_up_to_year]
            if len(latest_year_data) == 0:
                # if there is no data for the exact year, use the most recent year's data
                max_year = station_data['Year'].max()
                latest_year_data = station_data[station_data['Year'] == max_year]
            
            # initialize predictions for this station
            p90_prediction = None
            gm_prediction = None
            p90_accuracy = None
            gm_accuracy = None
            
            # P90 prediction
            if 'P90' in station_data.columns and not station_data['P90'].isnull().all():
                # prepare data
                X_p90 = station_data[feature_cols_p90].values
                y_p90 = station_data['P90'].values

                # create polynomial features (squared values)
                numeric_features = ['GM', 'SDV', 'MAX_']  
                X_enhanced_p90 = X_p90.copy()
                
                # add squared values 
                for feature in numeric_features:
                    if feature in feature_cols_p90:
                        i = feature_cols_p90.index(feature)
                        # create the squared values
                        squared_values = X_p90[:, i] ** 2
                        # add values to the enhanced matrix
                        X_enhanced_p90 = np.hstack((X_enhanced_p90, squared_values.reshape(-1, 1)))
                
                # scale features
                scaler_p90 = StandardScaler()
                X_scaled_p90 = scaler_p90.fit_transform(X_enhanced_p90)
                
                # convert to tensors
                X_tensor_p90 = torch.tensor(X_scaled_p90, dtype=torch.float32).to(device)
                y_tensor_p90 = torch.tensor(y_p90, dtype=torch.float32).view(-1, 1).to(device)
                
                # check if there is enough data for BatchNorm (need at least 2 samples)
                if len(station_data) < 2:
                    # use a simpler model without BatchNorm for stations that have less than 2 samples
                    class SimpleModel(nn.Module):
                        def __init__(self, input_size):
                            super().__init__()
                            self.network = nn.Sequential(
                                nn.Linear(input_size, 16),
                                nn.ReLU(),
                                nn.Linear(16, 1)
                            )

                        def forward(self, x):
                            return self.network(x)
                    
                    model_p90 = SimpleModel(X_enhanced_p90.shape[1]).to(device)
                    if station not in [s.split(':')[0] for s in limited_data_stations]:
                        limited_data_stations.append(f"{station}: Only {len(station_data)} records, using simple model") 
                else:
                    # use the standard model with BatchNorm for stations with more than 2 samples
                    model_p90 = StationModel(X_enhanced_p90.shape[1]).to(device)
                
                criterion = nn.MSELoss()
                optimizer = torch.optim.SGD(model_p90.parameters(), lr=0.01, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5, verbose=False)

                # early stopping variables for P90
                best_loss_p90 = float('inf')
                best_model_state_p90 = None
                patience = 15  
                patience_counter = 0

                # training loop for P90
                epochs = 50
                for epoch in range(epochs):
                    # training
                    model_p90.train()
                    
                    # forward pass
                    y_pred = model_p90(X_tensor_p90)
                    
                    # calculate loss
                    loss = criterion(y_pred, y_tensor_p90)
                    
                    # optimizer zero grad
                    optimizer.zero_grad()
                    
                    # backpropagation
                    loss.backward()
                    
                    # optimizer step
                    optimizer.step()
                    
                    # update learning rate
                    scheduler.step(loss)
                    
                    # early stopping logic
                    if loss.item() < best_loss_p90:
                        best_loss_p90 = loss.item()
                        best_model_state_p90 = model_p90.state_dict().copy()  
                        patience_counter = 0  
                    else:
                        patience_counter += 1  
                    
                    # if there are no improvement for 'patience' in alotted epochs, stop training
                    if patience_counter >= patience:
                        if station == stations[0]:
                            print(f"Early stopping for station {station} P90 model at epoch {epoch}")
                        break
                
                # after training, load the best model for prediction
                model_p90.eval()
                if best_model_state_p90 is not None:
                    model_p90.load_state_dict(best_model_state_p90)

                # get latest data for prediction (from the most recent year available)
                latest_data_p90 = latest_year_data.iloc[0][feature_cols_p90].values.reshape(1, -1)
                latest_data_p90_copy = latest_data_p90.copy()             

                # modify the Year value to be the prediction year
                year_index = feature_cols_p90.index('Year') if 'Year' in feature_cols_p90 else None
                if year_index is not None:
                    latest_data_p90_copy[0, year_index] = predict_for_year

                # add the squared values to the prediction data
                latest_data_enhanced_p90 = latest_data_p90_copy.copy()
                for feature in numeric_features:
                    if feature in feature_cols_p90:
                        i = feature_cols_p90.index(feature)
                        # create squared values
                        squared_value = latest_data_p90_copy[0, i] ** 2
                        # add to enhanced features
                        latest_data_enhanced_p90 = np.hstack((latest_data_enhanced_p90, np.array([[squared_value]])))

                # scale with the same scaler used in training
                latest_scaled_p90 = scaler_p90.transform(latest_data_enhanced_p90)
                X_test_p90 = torch.tensor(latest_scaled_p90, dtype=torch.float32).to(device)

                # make P90 prediction
                with torch.inference_mode():
                    p90_prediction = model_p90(X_test_p90).item()
                    
                    # forwards pass for accuracy
                    test_preds_p90 = model_p90(X_tensor_p90)
                    
                    # calculate accuracy
                    mse_metric = MeanSquaredError().to(device)
                    max_expected_mse = 100.0  
                    test_mse_p90 = mse_metric(test_preds_p90, y_tensor_p90)
                    p90_accuracy = 100 * (1 - min(test_mse_p90 / max_expected_mse, 1.0))
                    
                    p90_prediction = max(0, p90_prediction)  
                    p90_prediction = round(p90_prediction, 1)  

                if station == stations[0]:
                    print(f"Final P90 test accuracy for {station}: {p90_accuracy:.2f}%")

            # GM prediction
            if 'GM' in station_data.columns and not station_data['GM'].isnull().all():
                # prepare data
                X_gm = station_data[feature_cols_gm].values
                y_gm = station_data['GM'].values

                # create polynomial features (squared values)
                numeric_features = ['P90', 'SDV', 'MAX_']  
                X_enhanced_gm = X_gm.copy()
                
                # add squared values 
                for feature in numeric_features:
                    if feature in feature_cols_gm:
                        i = feature_cols_gm.index(feature)
                        # create the squared values
                        squared_values = X_gm[:, i] ** 2
                        # add values to the enhanced matrix
                        X_enhanced_gm = np.hstack((X_enhanced_gm, squared_values.reshape(-1, 1)))
                
                # scale features
                scaler_gm = StandardScaler()
                X_scaled_gm = scaler_gm.fit_transform(X_enhanced_gm)
                
                # convert to tensors
                X_tensor_gm = torch.tensor(X_scaled_gm, dtype=torch.float32).to(device)
                y_tensor_gm = torch.tensor(y_gm, dtype=torch.float32).view(-1, 1).to(device)
                
                # check if there is enough data for BatchNorm (need at least 2 samples)
                if len(station_data) < 2:
                    # use a simpler model without BatchNorm for stations that have less than 2 samples
                    class SimpleModel(nn.Module):
                        def __init__(self, input_size):
                            super().__init__()
                            self.network = nn.Sequential(
                                nn.Linear(input_size, 16),
                                nn.ReLU(),
                                nn.Linear(16, 1)
                            )

                        def forward(self, x):
                            return self.network(x)
                    
                    model_gm = SimpleModel(X_enhanced_gm.shape[1]).to(device)
                    if station not in [s.split(':')[0] for s in limited_data_stations]:
                        limited_data_stations.append(f"{station}: Only {len(station_data)} records, using simple model") 
                else:
                    # use the standard model with BatchNorm for stations with more than 2 samples
                    model_gm = StationModel(X_enhanced_gm.shape[1]).to(device)
                
                criterion = nn.MSELoss()
                optimizer = torch.optim.SGD(model_gm.parameters(), lr=0.01, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5, verbose=False)

                # early stopping variables for GM
                best_loss_gm = float('inf')
                best_model_state_gm = None
                patience = 15  
                patience_counter = 0

                # training loop for GM
                epochs = 50
                for epoch in range(epochs):
                    # training
                    model_gm.train()
                    
                    # forward pass
                    y_pred = model_gm(X_tensor_gm)
                    
                    # calculate loss
                    loss = criterion(y_pred, y_tensor_gm)
                    
                    # optimizer zero grad
                    optimizer.zero_grad()
                    
                    # backpropagation
                    loss.backward()
                    
                    # optimizer step
                    optimizer.step()
                    
                    # update learning rate
                    scheduler.step(loss)
                    
                    # early stopping logic
                    if loss.item() < best_loss_gm:
                        best_loss_gm = loss.item()
                        best_model_state_gm = model_gm.state_dict().copy()  
                        patience_counter = 0  
                    else:
                        patience_counter += 1  
                    
                    # if there are no improvement for 'patience' in alotted epochs, stop training
                    if patience_counter >= patience:
                        if station == stations[0]:
                            print(f"Early stopping for station {station} GM model at epoch {epoch}")
                        break
                
                # after training, load the best model for prediction
                model_gm.eval()
                if best_model_state_gm is not None:
                    model_gm.load_state_dict(best_model_state_gm)

                
                if p90_prediction is not None and 'P90' in feature_cols_gm:
                    # get latest data for prediction
                    latest_data_gm = latest_year_data.iloc[0][feature_cols_gm].values.reshape(1, -1)
                    latest_data_gm_copy = latest_data_gm.copy() 
                    
                    # update the P90 value with our prediction
                    p90_index = feature_cols_gm.index('P90')
                    latest_data_gm_copy[0, p90_index] = p90_prediction
                    
                    # update the Year value
                    year_index = feature_cols_gm.index('Year') if 'Year' in feature_cols_gm else None
                    if year_index is not None:
                        latest_data_gm_copy[0, year_index] = predict_for_year
                    
                    # add squared values
                    latest_data_enhanced_gm = latest_data_gm_copy.copy()
                    for feature in numeric_features:
                        if feature in feature_cols_gm:
                            i = feature_cols_gm.index(feature)
                            squared_value = latest_data_gm_copy[0, i] ** 2
                            latest_data_enhanced_gm = np.hstack((latest_data_enhanced_gm, np.array([[squared_value]])))
                    
                    # scale features
                    latest_scaled_gm = scaler_gm.transform(latest_data_enhanced_gm)
                    X_test_gm = torch.tensor(latest_scaled_gm, dtype=torch.float32).to(device)
                else:
                    # if P90 prediction not available, use the original data
                    latest_data_gm = latest_year_data.iloc[0][feature_cols_gm].values.reshape(1, -1)
                    latest_data_gm_copy = latest_data_gm.copy() 
                    
                    # update Year
                    year_index = feature_cols_gm.index('Year') if 'Year' in feature_cols_gm else None
                    if year_index is not None:
                        latest_data_gm_copy[0, year_index] = predict_for_year
                    
                    # add squared values
                    latest_data_enhanced_gm = latest_data_gm_copy.copy()
                    for feature in numeric_features:
                        if feature in feature_cols_gm:
                            i = feature_cols_gm.index(feature)
                            squared_value = latest_data_gm_copy[0, i] ** 2
                            latest_data_enhanced_gm = np.hstack((latest_data_enhanced_gm, np.array([[squared_value]])))
                    
                    # scale features
                    latest_scaled_gm = scaler_gm.transform(latest_data_enhanced_gm)
                    X_test_gm = torch.tensor(latest_scaled_gm, dtype=torch.float32).to(device)

                # make GM prediction
                with torch.inference_mode():
                    gm_prediction = model_gm(X_test_gm).item()
                    
                    # forward pass
                    test_preds_gm = model_gm(X_tensor_gm)
                    
                    # calculate accuracy
                    mse_metric = MeanSquaredError().to(device)
                    max_expected_mse = 100.0  
                    test_mse_gm = mse_metric(test_preds_gm, y_tensor_gm)
                    gm_accuracy = 100 * (1 - min(test_mse_gm / max_expected_mse, 1.0))
                    
                    gm_prediction = max(0, gm_prediction)  
                    gm_prediction = round(gm_prediction, 1)  

                if station == stations[0]:
                    print(f"Final GM test accuracy for {station}: {gm_accuracy:.2f}%")

            # create the result entry with both predictions
            result_entry = {
                'Station': station,
                'Year': predict_for_year
            }
            
            # add predictions 
            if p90_prediction is not None:
                result_entry['Predicted_P90'] = p90_prediction
                result_entry['P90_Model_Accuracy'] = float(p90_accuracy.cpu().numpy()) if isinstance(p90_accuracy, torch.Tensor) else p90_accuracy
            
            if gm_prediction is not None:
                result_entry['Predicted_GM'] = gm_prediction
                result_entry['GM_Model_Accuracy'] = float(gm_accuracy.cpu().numpy()) if isinstance(gm_accuracy, torch.Tensor) else gm_accuracy


            # add geographical coordinates
            if station in station_coords:
                result_entry['Lat_DD'] = station_coords[station][0]
                result_entry['Long_DD'] = station_coords[station][1]

            predictions.append(result_entry)
                
            # save the results as a temporary file every 50 stations
            if len(predictions) % 50 == 0:
                temp_df = pd.DataFrame(predictions)
                temp_df.to_csv('predictions_temp.csv', index=False)
        
        except Exception as e:
            error_stations.append(f"{station}: {str(e)}")
            print(f"Error processing station {station}: {e}")
    
    # create the result DataFrame
    results_df = pd.DataFrame(predictions)
    
    # calculate overall accuracy
    valid_p90_accuracies = results_df['P90_Model_Accuracy'].dropna() if 'P90_Model_Accuracy' in results_df.columns else pd.Series([])
    valid_gm_accuracies = results_df['GM_Model_Accuracy'].dropna() if 'GM_Model_Accuracy' in results_df.columns else pd.Series([])
    
    overall_p90_accuracy = valid_p90_accuracies.mean() if len(valid_p90_accuracies) > 0 else None
    overall_gm_accuracy = valid_gm_accuracies.mean() if len(valid_gm_accuracies) > 0 else None
    
    # print summary 
    print("\n===== SUMMARY =====")
    print(f"Total stations processed: {total_stations}")
    print(f"Successful predictions with neural network: {total_stations - len(limited_data_stations) - len(error_stations) - len(timeout_stations) - len(skipped_stations)}")
    print(f"Stations with limited data (using simpler model): {len(limited_data_stations)}")
    print(f"Stations with errors: {len(error_stations)}")
    print(f"Stations with timeouts: {len(timeout_stations)}")
    print(f"Stations skipped (no data): {len(skipped_stations)}")
    
    # print examples of stations with issues
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
    
    if overall_p90_accuracy is not None:
        print(f"\nOverall P90 model accuracy (across all stations): {overall_p90_accuracy:.2f}%")
    if overall_gm_accuracy is not None:
        print(f"\nOverall GM model accuracy (across all stations): {overall_gm_accuracy:.2f}%")
    
    # save to a .csv file
    output_file = f'p90_gm_predictions_{predict_for_year}_using_data_through_{train_up_to_year}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(results_df)} predictions to {output_file}")
    
    return results_df


###################
###### MODEL ######
###################

class StationModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)
    


###################
###### MAIN #######
###################

if __name__ == "__main__":

    # start timing
    start_time = time.time()
    
    print(f"\n=== Predicting P90 and GM values using data through given year ===")
    
    # process data
    train_data, stations, station_coords = process_data(train_up_to_year=2023)
    
    # train models and make predictions
    results = train_and_predict(train_data, 
                                stations, 
                                station_coords, 
                                train_up_to_year=2023, 
                                predict_for_year=2024)
    
    # overall time
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")