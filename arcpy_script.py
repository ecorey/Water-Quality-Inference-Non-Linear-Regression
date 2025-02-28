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
output_query_csv = r"C:\Users\bengs\Downloads\P90\Selected_Station_Query.csv"

# created point feature class for selected stations
output_point_fc = os.path.join(project_gdb, "Innacurate_Station_Points")

# set environment
arcpy.env.overwriteOutput = True


###################


# step 1: read the .csv using pandas
print("Reading the .csv file...")
df = pd.read_csv(nn_inference_file)

# replace NaN values in the 'note' column with 'No Note'
df['Note'] = df['Note'].fillna("No Note")

# save the .csv as a temporary file
nn_inference_file_modified = r"C:\Users\bengs\Downloads\P90\cleaned_p90_predictions.csv"
df.to_csv(nn_inference_file_modified, index=False)


###################


# step 2: create a table view from the .csv
print("Creating Table View from the .csv file...")
arcpy.management.MakeTableView(nn_inference_file_modified, "csv_view")


###################


# step 3: create a point feature class using the table view
print("Converting Cleaned CSV Table View to Point Feature Class...")
arcpy.management.XYTableToPoint("csv_view", 
                               os.path.join(project_gdb, inferred_p90_points_fc), 
                               "Long_DD", "Lat_DD", 
                               coordinate_system=arcpy.SpatialReference(4326))


###################


# step 4: spatial join with the new feature class (inferred values) and the target feature class (actual values)
print("Performing Spatial Join...")
spatial_join_output = os.path.join(project_gdb, spatial_join_inferred_and_actual)
arcpy.analysis.SpatialJoin(actual_p90_points_fc, 
                           os.path.join(project_gdb, inferred_p90_points_fc), 
                           spatial_join_output, 
                           join_type="KEEP_COMMON", 
                           match_option="CLOSEST")


###################


# Step 5: Create a new field called P90_DIFF and calculate the difference
print("Creating and Calculating P90_DIFF field...")
field_name = "P90_DIFF"
arcpy.management.AddField(spatial_join_output, field_name, "DOUBLE")
arcpy.management.CalculateField(spatial_join_output, 
                                field_name, 
                                "!P90! - !Predicted_P90!", 
                                "PYTHON3")


###################


# Step 6: Select Using Query and Export to CSV
print("Selecting with Query...")

# Build the query for selection
query = f"({field_name} > 10 OR {field_name} < -10) AND Model_Accuracy IS NOT NULL AND Note = 'No Note'"

# Directly select and create a new feature class without creating a layer on the map
arcpy.analysis.Select(spatial_join_output, output_point_fc, query)
print(f"New Point Feature Class created: {output_point_fc}")

# Count the selected stations
selected_count = int(arcpy.management.GetCount(output_point_fc)[0])
print(f"Number of stations matching query: {selected_count}")


###################


# Step 7: Export Selected Stations to CSV
print("Exporting Selected Stations to CSV...")

# Export the selected stations to a new CSV
arcpy.conversion.TableToTable(output_point_fc, 
                              os.path.dirname(output_query_csv), 
                              os.path.basename(output_query_csv))
print(f"Query result exported to: {output_query_csv}")

print("Process Completed Successfully!")



