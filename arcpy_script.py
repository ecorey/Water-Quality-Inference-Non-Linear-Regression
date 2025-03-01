"""
The goal of this script is to identify stations with significant differences between actual and predicted P90 values.

Workflow overview:
1. Reads a .csv file containing P90 predictions and modifies it.
2. Creates a table view from the .csv file.
3. Converts the table view to a point feature class.
4. Performs a spatial join between the actual and predicted P90 values.
5. Calculates the difference between actual and predicted P90 values.
6. Selects stations with significant differences and creates a new feature class.
7. Exports the selected stations to a new .csv file.
8. Confirms the completion of the process.
"""

import arcpy
import os
import pandas as pd


###################
##### PARAMS ######
###################

# path to the geodatabase
project_gdb = r"C:\Users\bengs\OneDrive\Documents\ArcGIS\Projects\P90\P90.gdb"  
# path to the nn prediction .csv file after nn is run
nn_inference_file = r"C:\Users\bengs\Downloads\P90\p90_predictions_2020_using_data_through_2019.csv" 
# created point feature class for actual values
actual_p90_points_fc = os.path.join(project_gdb, "c2020_P90_Scores_XYTableToPoint")   
# created point feature class for inferred values
inferred_p90_points_fc = "P90_Prediction_Points_From_CSV"     
# created spatial join layer for the inferred values and the actual values           
spatial_join_inferred_and_actual = "Spatial_Join_Actual_and_Prediction"   

# create .csv file for selected stations that are considered inaccurate
innacurate_stations_csv = r"C:\Users\bengs\Downloads\P90\Selected_Station_Query.csv"

# created point feature class for selected stations
innacurate_stations_point_fc = os.path.join(project_gdb, "Innacurate_Station_Points")

# set environment
arcpy.env.overwriteOutput = True


###################
##### STEP 1 ######
###################


# read the .csv with pandas
print("Reading the .csv file...")
df = pd.read_csv(nn_inference_file)



# save the .csv as a temporary file
nn_inference_file_modified = r"C:\Users\bengs\Downloads\P90\modified_p90_predictions.csv"
df.to_csv(nn_inference_file_modified, index=False)


###################
##### STEP 2 ######
###################


# create a table view from the .csv
print("Creating table view from the .csv file...")
arcpy.management.MakeTableView(nn_inference_file_modified, "csv_view")


###################
##### STEP 3 ######
###################


# create a point feature class using the table view
print("Creating point feature class for inferred P90 values...")
arcpy.management.XYTableToPoint("csv_view", 
                               os.path.join(project_gdb, inferred_p90_points_fc), 
                               "Long_DD", "Lat_DD", 
                               coordinate_system=arcpy.SpatialReference(4326))


###################
##### STEP 4 ######
###################


# spatial join with the new point feature class (inferred values) and the target point feature class (actual values)
print("Performing spatial join...")
spatial_join_output = os.path.join(project_gdb, spatial_join_inferred_and_actual)
arcpy.analysis.SpatialJoin(actual_p90_points_fc, 
                           os.path.join(project_gdb, inferred_p90_points_fc), 
                           spatial_join_output, 
                           join_type="KEEP_COMMON", 
                           match_option="CLOSEST")


###################
##### STEP 5 ######
###################


# create a new field called P90_DIFF and calculate the difference between the actual and predicted P90 values
print("Creating and calculating the P90_DIFF field...")
p90_diff_field_name = "P90_DIFF"
arcpy.management.AddField(spatial_join_output, p90_diff_field_name, "DOUBLE")
arcpy.management.CalculateField(spatial_join_output, 
                                p90_diff_field_name, 
                                "!P90! - !Predicted_P90!", 
                                "PYTHON3")


###################
##### STEP 6 ######
###################


# select the inaccurate stations with a query and create a new feature class 
print("Selecting the stations that are considered inaccurate...")

# query stations that are considered inaccurate
query = f"({p90_diff_field_name} > 13.9 OR {p90_diff_field_name} < -13.9) AND Model_Accuracy IS NOT NULL"

# select and create a new feature class with the selected stations
arcpy.analysis.Select(spatial_join_output, innacurate_stations_point_fc, query)
print(f"Inaccurate stations point feature class created: {innacurate_stations_point_fc}")

# count the selected stations
selected_stations_count = int(arcpy.management.GetCount(innacurate_stations_point_fc)[0])
print(f"Number of stations matching query: {selected_stations_count}")


###################
##### STEP 7 ######
###################


# create a new .csv file with the selected stations  
print("Creating selected stations .csv file...")

# create the .csv file from the selected stations   
arcpy.conversion.TableToTable(innacurate_stations_point_fc, 
                              os.path.dirname(innacurate_stations_csv), 
                              os.path.basename(innacurate_stations_csv))
print(f"Inaccurate stations query results exported to: {innacurate_stations_csv}")


###################
##### STEP 8 ######
###################


# confirm completion
print("Process Completed Successfully!")



