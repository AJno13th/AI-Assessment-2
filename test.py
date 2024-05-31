# Import the pandas library and give it the alias 'pd'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as Matplot  # Use Matplot for plotting

# Step 1: Read the data
# Define the file path to the dataset. The 'r' prefix is used to handle the backslashes in the Windows file path correctly.
file_path = r'C:\\Users\\antho\\Downloads\\Dataset of Diabetes .csv'  # Update the path as needed

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

# Convert categorical columns to numerical (if needed)
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})

# Convert target variable 'CLASS' to numerical
if 'CLASS' in data.columns:
    data['CLASS'] = data['CLASS'].map({'N': 0, 'Y': 1, 'P': 2})  # Assuming 'Y' and 'P' are other classes

# Remove rows with NaN values in 'CLASS' after mapping
data = data.dropna(subset=['CLASS'])

# Features and target split
X = data.drop('CLASS', axis=1)
y = data['CLASS']

# Check for specific column names and handle missing columns
if 'BMI' in X.columns and 'AGE' in X.columns:
    X['BMI_Age'] = X['BMI'] * X['AGE']
else:
    print("Columns 'BMI' or 'AGE' not found in the dataset.")
    # If columns are missing, create BMI_Age with default values (optional)
    X['BMI_Age'] = np.zeros(len(X))

# Ensure no NaN values in features
X = X.fillna(0)
print("Missing values in features after filling:", X.isnull().sum().sum())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Building and training the model
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
training_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, verbose=1)

# Step 4: Evaluating the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:\n", cm)

# Step 7: Visualizing the training process
# Plot training & validation accuracy values
Matplot.plot(training_history.history['accuracy'])
Matplot.plot(training_history.history['val_accuracy'])
Matplot.title('Model accuracy')
Matplot.ylabel('Accuracy')
Matplot.xlabel('Epoch')
Matplot.legend(['Train', 'Validation'], loc='upper left')
Matplot.show()

# Plot training & validation loss values
Matplot.plot(training_history.history['loss'])
Matplot.plot(training_history.history['val_loss'])
Matplot.title('Model loss')
Matplot.ylabel('Loss')
Matplot.xlabel('Epoch')
Matplot.legend(['Train', 'Validation'], loc='upper left')
Matplot.show()
