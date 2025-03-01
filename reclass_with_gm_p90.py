import pandas as pd

def classify_stations(input_csv, output_csv=None):
    """
    Reclassify stations in a CSV file based on predicted GM and P90 values.
    
    Args:
        input_csv: Path to the input CSV file with predictions
        output_csv: Path to save the output CSV file. If None, overwrites the input
    
    Classification standards:
        - Approved (A): GM ≤ 14 and P90 ≤ 31
        - Restricted (R): GM ≤ 88 and P90 ≤ 163 (but not meeting Approved criteria)
        - Prohibited (P): GM > 88 or P90 > 163
    """
    # Read the CSV file
    print(f"Reading predictions from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Check if required columns exist
    required_cols = ['Predicted_GM', 'Predicted_P90']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV file")
    
    # Make a copy of the original Class column if it exists
    if 'Class' in df.columns:
        df['Original_Class'] = df['Class']
    
    # Define a function to classify based on GM and P90 values
    def get_classification(row):
        gm = row['Predicted_GM']
        p90 = row['Predicted_P90']
        
        if gm <= 14 and p90 <= 31:
            return 'A'  # Approved
        elif gm <= 88 and p90 <= 163:
            return 'R'  # Restricted
        else:
            return 'P'  # Prohibited
    
    # Apply the classification function
    df['Class'] = df.apply(get_classification, axis=1)
    
    # Count the number of stations in each class
    class_counts = df['Class'].value_counts()
    print("\nClass distribution after reclassification:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} stations")
    
    # Compare with original classification if available
    if 'Original_Class' in df.columns:
        # Count stations that changed classification
        df['Changed'] = df['Class'] != df['Original_Class']
        changed_count = df['Changed'].sum()
        
        print(f"\nStations that changed classification: {changed_count} ({changed_count/len(df)*100:.1f}%)")
        
        # Show most common changes
        if changed_count > 0:
            changes = df[df['Changed']].groupby(['Original_Class', 'Class']).size().reset_index(name='Count')
            changes = changes.sort_values('Count', ascending=False)
            print("\nMost common classification changes (original → new):")
            for _, row in changes.head(5).iterrows():
                print(f"  {row['Original_Class']} → {row['Class']}: {row['Count']} stations")
        
        # Remove the temporary column
        df = df.drop(columns=['Changed', 'Original_Class'])
    
    # Save the results
    if output_csv is None:
        output_csv = input_csv
    
    df.to_csv(output_csv, index=False)
    print(f"\nSaved reclassified data to {output_csv}")
    
    return df

if __name__ == "__main__":
    # Example usage
    input_file = "/home/ub/P90/p90_gm_predictions_2024_using_data_through_2023.csv"
    classify_stations(input_file)