# Import the pandas library and give it the alias 'pd'
import pandas as pd

# Step 1: Read the data
# Define the file path to the dataset. The 'r' prefix is used to handle the backslashes in the Windows file path correctly.
file_path = r'C:\Users\antho\Downloads\Dataset of Diabetes .csv'  #can be updated to search for the csv file within the folder

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Print column names for verification
# This helps to ensure the data has been loaded correctly and to see the names of all columns in the dataset.
print(data.columns)

# Step 2: Data cleaning and preprocessing
# Fill NaN values in 'Gender' with the mode (most frequent value)
# Check if the 'Gender' column exists in the DataFrame
if 'Gender' in data.columns:
    # Calculate the mode of the 'Gender' column
    gender_mode = data['Gender'].mode()[0]
    # Fill NaN values in the 'Gender' column with the mode value
    data['Gender'].fillna(gender_mode, inplace=True)

# Fill NaN values in other columns with the mean of the respective column
# Select columns with numeric data types
numeric_columns = data.select_dtypes(include=[np.number]).columns
# Iterate through each numeric column
for col in numeric_columns:
    # Fill NaN values in the column with the mean of the column
    data[col].fillna(data[col].mean(), inplace=True)

# Verify that there are no NaN values left
# Calculate and print the total number of missing values in the entire DataFrame
print("Missing values after filling:", data.isnull().sum().sum())
