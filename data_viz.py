import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

# Load and combine all datasets
dataframes = []
for file in file_list:
    try:
        year = int(file.split('_')[0].split('/')[-1])
        df = pd.read_csv(file)
        
        # Replace any infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Add year column
        df['Year'] = year
        dataframes.append(df)
        print(f"Loaded {file} with {len(df)} rows")
    except Exception as e:
        print(f"Could not load {file}: {e}")

# Combine all data
all_data = pd.concat(dataframes, ignore_index=True)
print(f"Combined dataset has {len(all_data)} rows")

# Get the first 5 unique stations
first_5_stations = all_data['Station'].unique()[2:7]
print(f"First 5 stations for visualization: {first_5_stations}")

# Filter data for just these stations
selected_data = all_data[all_data['Station'].isin(first_5_stations)]

# Create a scatter plot for P90 values across years
plt.figure(figsize=(12, 8))

# Define colors for each station
colors = ['blue', 'red', 'green', 'purple', 'orange']

# Plot each station with a different color
for i, station in enumerate(first_5_stations):
    station_data = selected_data[selected_data['Station'] == station]
    
    # Sort by year for consistent plotting
    station_data = station_data.sort_values('Year')
    
    # Plot scatter points
    plt.scatter(station_data['Year'], station_data['P90'], 
                color=colors[i], label=station, alpha=0.7, s=80)
    
    # Add connecting lines
    plt.plot(station_data['Year'], station_data['P90'], 
             color=colors[i], alpha=0.5, linestyle='-')

# Add labels and title
plt.xlabel('Year', fontsize=14)
plt.ylabel('P90 Value', fontsize=14)
plt.title('P90 Values for First 5 Stations (2013-2022)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Station ID')

# Adjust x-axis to show all years
plt.xticks(range(2013, 2023))

# Add some statistics in text boxes
for i, station in enumerate(first_5_stations):
    station_data = selected_data[selected_data['Station'] == station]
    
    # Calculate statistics
    mean_p90 = station_data['P90'].mean()
    
    # Add text annotation for the mean
    plt.annotate(f'Mean: {mean_p90:.1f}', 
                 xy=(2022.1, station_data[station_data['Year'] == station_data['Year'].max()]['P90'].values[0]),
                 color=colors[i],
                 fontsize=10)

# Save the figure
plt.tight_layout()
plt.savefig('p90_station_visualization.png', dpi=300)
print("Saved visualization to p90_station_visualization.png")

# Display the figure
plt.show()

# Print simple table of the data for these stations
print("\nP90 values by station and year:")
pivot_table = selected_data.pivot_table(values='P90', index='Station', columns='Year', aggfunc='mean')
print(pivot_table)