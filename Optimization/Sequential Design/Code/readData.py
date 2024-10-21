import os
import pandas as pd

# Set the directory containing the CSV files
folder_path = 'D:/ResearchProject/Optimization/Python/Sequential/Results'  # Update this path to your folder containing CSV files

# Initialize an empty DataFrame to store the combined data
Results = pd.DataFrame()

# Iterate over the files in the directory
for file_name in os.listdir(folder_path):
    # Check if the file is a CSV file
    if file_name.endswith('.csv'):
        # Read the CSV file into a DataFrame
        file_path = os.path.join(folder_path, file_name)
        temp_df = pd.read_csv(file_path)
        temp_df.rename(
            index={temp_df.index[0]: file_name.replace('DF_', '').replace('result', 'Calibration').replace('.csv', '')},
            inplace=True)
        Results = pd.concat([Results, temp_df], axis=0)

# Save the combined DataFrame to a new CSV file
output_file_path = os.path.join(folder_path.replace("/Sequential/Results", ""), 'Results.csv')
Results.to_csv(output_file_path, index=True)