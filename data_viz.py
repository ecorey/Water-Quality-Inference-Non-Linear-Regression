import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# paths to the P90 score files
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
    './data/2022_P90_Scores.csv',
    './data/2023_P90_Scores.csv'
]

# load the data from each file into a list of dataframes
dataframes = []
for file in file_list:
    try:
        year = int(file.split('_')[0].split('/')[-1])
        df = pd.read_csv(file)
        
        # replace values with infinite with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # add a year column
        df['Year'] = year
        dataframes.append(df)
        print(f"Loaded {file} with {len(df)} rows")
    except Exception as e:
        print(f"Could not load {file}: {e}")


# combine all data for the 10 years into a single dataframe so that we can visualize it as a line plot
all_data = pd.concat(dataframes, ignore_index=True)
print(f"Combined dataset has {len(all_data)} rows")

# get 5 stations to visualize the P90 values over time
five_stations = all_data['Station'].unique()[1240:1245]
print(f"First 5 stations for visualization: {five_stations}")

# filter the data for just these stations
selected_data = all_data[all_data['Station'].isin(five_stations)]

# create a scatter plot for the P90 values across years
plt.figure(figsize=(12, 8))

# Add background color shading for the classification thresholds
# P90 standards: Approved ≤ 31, Restricted ≤ 163, Prohibited > 163
# Get the x-axis range
x_min, x_max = 2013, 2023

# Add background colors with 75% opacity (alpha=0.75)
plt.axhspan(0, 31, facecolor='green', alpha=0.75, zorder=0)  # Approved
plt.axhspan(31, 163, facecolor='orange', alpha=0.75, zorder=0)  # Restricted
plt.axhspan(163, plt.ylim()[1] * 2, facecolor='red', alpha=0.75, zorder=0)  # Prohibited

# Add horizontal lines at the threshold boundaries
plt.axhline(y=31, color='black', linestyle='--', alpha=0.5, zorder=1)
plt.axhline(y=163, color='black', linestyle='--', alpha=0.5, zorder=1)

# colors for each station
colors = ['red', 'orange', 'yellow', 'green', 'blue']

# plot each station with a different color
for i, station in enumerate(five_stations):
    station_data = selected_data[selected_data['Station'] == station]
    
    # sort by year 
    station_data = station_data.sort_values('Year')
    
    # plot the scatter points
    plt.scatter(station_data['Year'], station_data['P90'], 
                color=colors[i], label=station, alpha=0.7, s=80)
    
    # add connecting lines for each station
    plt.plot(station_data['Year'], station_data['P90'], 
             color=colors[i], alpha=0.5, linestyle='-')

# add labels and a title
plt.xlabel('Year', fontsize=14)
plt.ylabel('P90 Value', fontsize=14)
plt.title('P90 Values for 5 Stations (2013-2023)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Station ID')

# Add threshold legend
plt.text(2013.5, 15, 'Approved (≤31)', fontsize=10, bbox=dict(facecolor='green', alpha=0.75))
plt.text(2013.5, 97, 'Restricted (≤163)', fontsize=10, bbox=dict(facecolor='orange', alpha=0.75))
plt.text(2013.5, 200, 'Prohibited (>163)', fontsize=10, bbox=dict(facecolor='red', alpha=0.75))

# adjust the x-axis to show all years
plt.xticks(range(2013, 2023))

# save the figure
plt.tight_layout()
plt.savefig('p90_station_visualization.png', dpi=300)
print("Saved visualization!")

# display the figure
plt.show()

# print a table of the data for these stations
print("\nP90 values by station and year:")
pivot_table = selected_data.pivot_table(values='P90', index='Station', columns='Year', aggfunc='mean')
print(pivot_table)