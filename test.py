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
import matplotlib.pyplot as plt

# Step 1: Read the data
# Define the file path to the dataset
dataset_path = r'C:\Users\antho\Downloads\Dataset of Diabetes .csv'  # Update the path as needed

# Read the CSV file into a pandas DataFrame
diabetes_data = pd.read_csv(dataset_path)

# Print column names for verification
print(diabetes_data.columns)

# Step 2: Data cleaning and preprocessing
# Fill NaN values in 'Gender' with the mode (most frequent value)
if 'Gender' in diabetes_data.columns:
    mode_gender = diabetes_data['Gender'].mode()[0]
    diabetes_data['Gender'].fillna(mode_gender, inplace=True)

# Fill NaN values in other columns with the mean of the respective column
num_cols = diabetes_data.select_dtypes(include=[np.number]).columns
for column in num_cols:
    diabetes_data[column].fillna(diabetes_data[column].mean(), inplace=True)

# Verify that there are no NaN values left
print("Missing values after filling:", diabetes_data.isnull().sum().sum())

# Convert categorical columns to numerical values
categorical_columns = diabetes_data.select_dtypes(include=[object]).columns
for column in categorical_columns:
    diabetes_data[column] = pd.factorize(diabetes_data[column])[0]

# Remove rows with NaN values in 'CLASS' column
diabetes_data.dropna(subset=['CLASS'], inplace=True)

# Create new feature 'BMI_Age'
diabetes_data['BMI_Age'] = diabetes_data['BMI'] * diabetes_data['AGE']

# Step 3: Feature Scaling
features = diabetes_data.drop(columns=['CLASS'])
target = diabetes_data['CLASS']

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Step 4: Hyperparameter Tuning
def build_ann(optimizer='adam'):
    ann_model = Sequential()
    ann_model.add(Input(shape=(train_features.shape[1],)))
    ann_model.add(Dense(units=16, activation='relu'))
    ann_model.add(Dense(units=8, activation='relu'))
    ann_model.add(Dense(units=1, activation='sigmoid'))
    ann_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return ann_model

ann = KerasClassifier(model=build_ann, verbose=0)

hyperparameters = {
    'batch_size': [10, 20],
    'epochs': [50, 100],
    'optimizer': ['adam', 'rmsprop']
}

grid_search_cv = GridSearchCV(estimator=ann, param_grid=hyperparameters, cv=StratifiedKFold(n_splits=5))
grid_search_cv.fit(train_features, train_target)

optimal_params = grid_search_cv.best_params_
print("Best Hyperparameters:", optimal_params)

# Step 5: Cross-Validation
cv_scores = cross_val_score(grid_search_cv, features, target, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Step 6: Model Training and Evaluation
optimal_model = build_ann(optimizer=optimal_params['optimizer'])
training_history = optimal_model.fit(train_features, train_target, batch_size=optimal_params['batch_size'], epochs=optimal_params['epochs'], validation_split=0.2, verbose=0)

test_features_scaled = scaler.transform(test_features)
pred_target = (optimal_model.predict(test_features_scaled) > 0.5).astype(int)

conf_matrix = confusion_matrix(test_target, pred_target)
acc_score = accuracy_score(test_target, pred_target)
prec_score = precision_score(test_target, pred_target, average='weighted')
f1score = f1_score(test_target, pred_target, average='weighted')
roc_auc_score = roc_auc_score(test_target, optimal_model.predict_proba(test_features_scaled), multi_class='ovr')

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", acc_score)
print("Precision:", prec_score)
print("F1 Score:", f1score)
print("ROC-AUC Score:", roc_auc_score)

# Step 7: Visualizing the training process
plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

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

7. Visualization:
   - Visualized the training process by plotting accuracy and loss over epochs for both training and validation sets.
"""

print(documentation)
