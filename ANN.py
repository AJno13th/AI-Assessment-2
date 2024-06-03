# Import the pandas library and give it the alias 'pd'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, roc_auc_score
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as matplot

# Step 1: Read the data
# Define the file path to the dataset. The 'r' prefix is used to handle the backslashes in the Windows file path correctly.
file_path = r'C:\Users\antho\Downloads\Dataset of Diabetes .csv'  # Update the path as needed

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

# Define a function to create the model, required for KerasClassifier
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Input(shape=(X_scaled.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Assuming three classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Step 3: Hyperparameter Tuning using GridSearchCV
model = KerasClassifier(model=create_model, verbose=0)
param_grid = {
    'batch_size': [10, 20],
    'epochs': [50, 100],
    'optimizer': ['adam', 'rmsprop']
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_scaled, y)

# Summarize results of hyperparameter tuning
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Step 4: Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_val_score(grid.best_estimator_, X_scaled, y, cv=kfold)
print("Cross-validation results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Step 5: Training the model with best parameters
best_model = grid.best_estimator_
history = best_model.fit(X_scaled, y, validation_split=0.2, epochs=grid_result.best_params_['epochs'], batch_size=grid_result.best_params_['batch_size'])

# Step 6: Evaluating the model
y_pred = best_model.predict(X_scaled)
cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')
roc_auc = roc_auc_score(y, best_model.predict_proba(X_scaled), multi_class='ovr')

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)
print("ROC-AUC Score:", roc_auc)

# Step : Save the model to the same folder as the script
best_model.model_.save('Best_model.h5')

# Step 7: Visualizing the training process
matplot.plot(history.history_['accuracy'])
matplot.plot(history.history_['val_accuracy'])
matplot.title('Model accuracy')
matplot.ylabel('Accuracy')
matplot.xlabel('Epoch')
matplot.legend(['Train', 'Validation'], loc='upper left')
matplot.show()

matplot.plot(history.history_['loss'])
matplot.plot(history.history_['val_loss'])
matplot.title('Model loss')
matplot.ylabel('Loss')
matplot.xlabel('Epoch')
matplot.legend(['Train', 'Validation'], loc='upper left')
matplot.show()

# Documenting the steps
documentation = """
1. Data Reading and Cleaning:
   - Read the dataset from the provided CSV file.
   - Filled missing values in 'Gender' with the mode and in numeric columns with the mean.
   - Converted categorical columns to numerical values.
   - Removed rows with NaN values in 'CLASS' column.
   - Created new feature 'BMI_Age' by multiplying 'BMI' and 'AGE'.

2. Feature Scaling:
   - Standardized the features using StandardScaler.

3. Hyperparameter Tuning:
   - Used GridSearchCV to find the best parameters for batch size, epochs, and optimizer.

4. Cross-Validation:
   - Implemented 5-fold cross-validation to ensure the model generalizes well to unseen data.

5. Model Training:
   - Trained the final model with the best parameters from GridSearchCV.

6. Model Evaluation:
   - Evaluated the model using confusion matrix, accuracy, precision, F1 score, and ROC-AUC score.

7. Model Save:
    - Saves the best ANN model to the same directory the script is saved in.

7. Visualization:
   - Visualized the training process by plotting accuracy and loss over epochs for both training and validation sets.
"""

print(documentation)
