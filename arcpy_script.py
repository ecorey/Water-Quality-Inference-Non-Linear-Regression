import arcpy
import os

# Parameters - Change these as needed
project_gdb = r"C:\Users\bengs\OneDrive\Documents\ArcGIS\Projects\P90\P90.gdb"  # Path to your geodatabase
csv_file = r"C:\Users\bengs\Downloads\P90\p90_predictions_2023_using_data_through_2022.csv"   # Path to your .csv file
output_fc = "P90_Prediction_Points_From_CSV"                 # Name of the new feature class
spatial_join_fc = "Spatial_Join_Actual_and_Prediction"        # Name of the spatial join output
target_fc = os.path.join(project_gdb, "c2023_P90_Scores_XYTableToPoint")  # Target feature class for spatial join
output_csv = r"C:\Users\bengs\Downloads\P90\Selected_P90_Diff.csv"  # Path to save the output CSV

# Set environment
arcpy.env.overwriteOutput = True

# 1. Create a point feature class using long_DD and lat_DD fields
print("Converting CSV to Point Feature Class...")
arcpy.management.XYTableToPoint(csv_file, 
                               os.path.join(project_gdb, output_fc), 
                               "long_DD", "lat_DD", 
                               coordinate_system=arcpy.SpatialReference(4326))

# 2. Perform a spatial join with the new feature class and 2023_P90_Scores_Actual
print("Performing Spatial Join...")
spatial_join_output = os.path.join(project_gdb, spatial_join_fc)
arcpy.analysis.SpatialJoin(target_fc, 
                           os.path.join(project_gdb, output_fc), 
                           spatial_join_output, 
                           join_type="KEEP_COMMON", 
                           match_option="CLOSEST")

# 3. Create a new field called P90_DIFF and calculate the difference
print("Creating and Calculating P90_DIFF field...")
field_name = "P90_DIFF"
arcpy.management.AddField(spatial_join_output, field_name, "DOUBLE")
arcpy.management.CalculateField(spatial_join_output, 
                                field_name, 
                                "!P90! - !Predicted_P90!", 
                                "PYTHON3")

# 4. Select and Count Stations with P90_DIFF > +8 or < -8
print("Selecting P90_DIFF outside of ±8...")
selection_query = f"{field_name} > 8 OR {field_name} < -8"
arcpy.management.MakeFeatureLayer(spatial_join_output, "Selected_Layer")
arcpy.management.SelectLayerByAttribute("Selected_Layer", 
                                        "NEW_SELECTION", 
                                        selection_query)

# Count the selected stations
selected_count = int(arcpy.management.GetCount("Selected_Layer")[0])
print(f"Number of stations with P90_DIFF > +8 or < -8: {selected_count}")

# 5. Export the selected points to a CSV file
print("Exporting selected points to CSV...")
arcpy.conversion.TableToTable("Selected_Layer", 
                              os.path.dirname(output_csv), 
                              os.path.basename(output_csv))

print(f"Selected points exported to: {output_csv}")
print("Process Completed Successfully!")
