import pandas as pd
import numpy as np
import ast
import re
import os

# def process_csv(input_file, output_file):
#     # Read the CSV
#     df = pd.read_csv(input_file)
    
#     # Replace the matrix strings with their trace
#     def compute_trace(matrix_string):
#         # Ensure the matrix string is properly formatted
#         # matrix_string = matrix_string.replace(' ', ',')  # Replace spaces with commas
#         print(matrix_string)
#         m = np.array(matrix_string)
#         print(m)
#         print(m.shape)
#         print(m[0][0])
#         matrix = np.array(ast.literal_eval(matrix_string))  # Convert string to numpy array
#         return np.trace(matrix)  # Compute trace of the matrix
    
#     df['goal_uncertainty'] = df['goal_uncertainty'].apply(compute_trace)
    
#     # Save the modified DataFrame to a new CSV
#     df.to_csv(output_file, index=False)
#     print(f"Processed file saved to {output_file}")

# def parse_and_compute_trace(matrix_string):
#     # use regex to extract numeric values from the matrix string
#     values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", matrix_string)))
#     # reshape the flat list into a square matrix assuming it's 2x2
#     matrix_size = int(len(values) ** 0.5)
#     matrix = np.array(values).reshape((matrix_size, matrix_size))
#     return np.trace(matrix)

# def process_csv(input_file, output_file):
#     # read the csv
#     df = pd.read_csv(input_file)
    
#     # replace the matrix strings with their trace
#     df['goal_uncertainty'] = df['goal_uncertainty'].apply(parse_and_compute_trace)
    
#     # save the modified dataframe to a new csv
#     df.to_csv(output_file, index=False)
#     print(f"processed file saved to {output_file}")
# # Example usage
# input_csv = "results/500_samples/BRM_1_1_results.csv"  # Replace with your input file path
# output_csv = "output.csv"  # Replace with your desired output file path
# process_csv(input_csv, output_csv)
import pandas as pd
import numpy as np

# def summarize_csv_files(file_list, output_csv):
#     """
#     Processes a list of CSV files and returns summary statistics.
    
#     Args:
#         file_list (list): List of file paths to CSV files.
    
#     Returns:
#         pd.DataFrame: DataFrame containing mean, median, min, and max for time, path_length, and goal_uncertainty.
#     """
#     summary_data = []

#     for file in file_list:
#         # Read the CSV file
#         filepath = os.path.join("results", file)
#         df = pd.read_csv(filepath)
        
#         # Compute summary statistics
#         stats = {
#             "file_name": file,
#             "time_mean": df["time"].mean(),
#             "time_median": df["time"].median(),
#             "time_min": df["time"].min(),
#             "time_max": df["time"].max(),
#             "path_length_mean": df["path_length"].mean(),
#             "path_length_median": df["path_length"].median(),
#             "path_length_min": df["path_length"].min(),
#             "path_length_max": df["path_length"].max(),
#             "goal_uncertainty_mean": df["goal_uncertainty"].mean(),
#             "goal_uncertainty_median": df["goal_uncertainty"].median(),
#             "goal_uncertainty_min": df["goal_uncertainty"].min(),
#             "goal_uncertainty_max": df["goal_uncertainty"].max(),
#         }
        
#         # Append to summary list
#         summary_data.append(stats)

#     # Convert the summary data into a DataFrame
#     summary_df = pd.DataFrame(summary_data)
#     # Save the summary DataFrame to a CSV file
#     summary_df.to_csv(output_csv, index=False)
#     print(f"Summary saved to {output_csv}")

# # Example usage
# file_list = ["100_BRM_1_1_results.csv", 
#              "100_BRM_1_2_results.csv",
#              "500_BRM_1_1_results.csv", 
#              "500_BRM_1_2_results.csv", 
#              "RRBT_1_1_results.csv", 
#              "RRBT_1_2_results.csv"]  # Replace with your actual file paths
# output_csv = "summary.csv"  # Replace with your desired output file path
# summarize_csv_files(file_list, output_csv)
df = pd.read_csv("summary.csv")

# Set the "Filename" column as the index and transpose the DataFrame
flipped_df = df.set_index("file_name").transpose()

# Save the flipped DataFrame to a new CSV file
flipped_df.to_csv("flipped_results.csv")