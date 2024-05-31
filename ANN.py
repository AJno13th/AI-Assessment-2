# Import the pandas library and give it the alias 'pd'
import pandas as pd
import numpy as np

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

# Step 3: Data Standardization
from sklearn.preprocessing import StandardScaler


# Standardize numerical features to ensure all features have a similar scale
numeric_columns = data.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Print the first few rows of the standardized numerical features for verification
print(data[numeric_columns].head())

# Step 4: Split the data into training and testing sets
from sklearn.model_selection import train_test_split


# Define features (X) and target (y)
X = data.drop(columns=['CLASS'])
y = data['CLASS']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets for verification
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Step 5: Model Creation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create the Artificial Neural Network (ANN) model
model = Sequential()

# Adding the input layer and the first hidden layer with 12 neurons and 'relu' activation function
model.add(Dense(units=12, activation='relu', input_dim=X_train.shape[1]))

# Adding the second hidden layer with 8 neurons and 'relu' activation function
model.add(Dense(units=8, activation='relu'))

# Adding the output layer with 1 neuron and 'sigmoid' activation function since it's a binary classification problem
model.add(Dense(units=1, activation='sigmoid'))

# Print the model summary to verify the architecture
print(model.summary())

# Step 6: Model Compilation
# Compile the ANN model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Model Training
# Train the ANN model on the training data
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# Step 8: Model Evaluation
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
