import arcpy
import os
import pandas as pd

# Parameters - Change these as needed
project_gdb = r"C:\Users\bengs\OneDrive\Documents\ArcGIS\Projects\P90\P90.gdb"  # Path to your geodatabase
csv_file = r"C:\Users\bengs\Downloads\P90\p90_predictions_2020_using_data_through_2019.csv"   # Path to your .csv file
output_fc = "P90_Prediction_Points_From_CSV"                 # Name of the new feature class
spatial_join_fc = "Spatial_Join_Actual_and_Prediction"        # Name of the spatial join output
target_fc = os.path.join(project_gdb, "c2020_P90_Scores_XYTableToPoint")  # Target feature class for spatial join

# Output CSVs
output_selected_csv = r"C:\Users\bengs\Downloads\P90\Selected_P90_Diff.csv"
output_query_csv = r"C:\Users\bengs\Downloads\P90\Selected_Station_Query.csv"

# Output Point Feature Class
output_point_fc = os.path.join(project_gdb, "Selected_Station_Points")

# Set environment
arcpy.env.overwriteOutput = True

# Step 1: Clean the CSV using pandas
print("Cleaning CSV file...")
df = pd.read_csv(csv_file)

# Replace NaN values in the 'Note' column with a placeholder
df['Note'] = df['Note'].fillna("No Note")

# Save cleaned CSV to a temporary file
cleaned_csv = r"C:\Users\bengs\Downloads\P90\cleaned_p90_predictions.csv"
df.to_csv(cleaned_csv, index=False)

# Step 2: Create a Table View from Cleaned CSV
print("Creating Table View from Cleaned CSV...")
arcpy.management.MakeTableView(cleaned_csv, "csv_view")

# Inspect fields to confirm "Note" field is present
fields = [f.name for f in arcpy.ListFields("csv_view")]
print("Fields in Cleaned CSV:", fields)

# Step 3: Create Point Feature Class using the Cleaned Table View
print("Converting Cleaned CSV Table View to Point Feature Class...")
arcpy.management.XYTableToPoint("csv_view", 
                               os.path.join(project_gdb, output_fc), 
                               "Long_DD", "Lat_DD", 
                               coordinate_system=arcpy.SpatialReference(4326))

# Step 4: Perform a Spatial Join with the new feature class and target FC
print("Performing Spatial Join...")
spatial_join_output = os.path.join(project_gdb, spatial_join_fc)
arcpy.analysis.SpatialJoin(target_fc, 
                           os.path.join(project_gdb, output_fc), 
                           spatial_join_output, 
                           join_type="KEEP_COMMON", 
                           match_option="CLOSEST")

# Step 5: Create a new field called P90_DIFF and calculate the difference
print("Creating and Calculating P90_DIFF field...")
field_name = "P90_DIFF"
arcpy.management.AddField(spatial_join_output, field_name, "DOUBLE")
arcpy.management.CalculateField(spatial_join_output, 
                                field_name, 
                                "!P90! - !Predicted_P90!", 
                                "PYTHON3")

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

# Step 7: Export Selected Stations to CSV
print("Exporting Selected Stations to CSV...")

# Export the selected stations to a new CSV
arcpy.conversion.TableToTable(output_point_fc, 
                              os.path.dirname(output_query_csv), 
                              os.path.basename(output_query_csv))
print(f"Query result exported to: {output_query_csv}")

print("Process Completed Successfully!")
