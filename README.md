NAME
    ANN.py - Diabetes AI training project

SYNOPSIS
    python ANN.py [OPTIONS]

DESCRIPTION
    ANN.py is a Python script that reads a dataset, preprocesses it, and trains an artificial neural network (ANN) to predict diabetes. The script includes data cleaning, preprocessing, model training, and evaluation steps.

OPTIONS
    -h, --help
        Show this help message and exit.

    -f FILE, --file FILE
        Specify the path to the dataset CSV file. This is a required option.

    -e EPOCHS, --epochs EPOCHS
        Specify the number of epochs for training the neural network. Default is 50.
        
    -b BATCH_SIZE, --batch_size BATCH_SIZE
        Specify the batch size for training the neural network. Default is 10.

    -o OPTIMIZER, --optimizer OPTIMIZER
        Specify the optimizer for training the neural network. Default is 'adam'.

    -v, --verbose
        Enable verbose output during script execution.

EXAMPLES
    python ANN.py -f path/to/dataset.csv -e 100 -b 20 -o 'sgd'

AUTHOR
    Anthony Joshua
    Canterbury Christ Church University
    A.Joshua483@csnterbury.ac.uk

REPORTING BUGS
    Report bugs to A.Joshua483@csnterbury.ac.uk 

COPYRIGHT
    This is free and open-source software.

DETAILS
    The script performs the following steps:

    1. Data Reading and Cleaning:
       - Read the dataset from the provided CSV file.
       - Fill missing values in 'Gender' with the mode and in numeric columns with the mean.
       - Convert categorical columns to numerical values.
       - Remove rows with NaN values in 'CLASS' column.
       - Create a new feature 'BMI_Age' by multiplying 'BMI' and 'AGE'.

    2. Feature Scaling:
       - Standardize the features using StandardScaler.

    3. Hyperparameter Tuning:
       - Use GridSearchCV to find the best parameters for batch size, epochs, and optimizer.

    4. Cross-Validation:
       - Implement 5-fold cross-validation to ensure the model generalizes well to unseen data.

    5. Model Training:
       - Train the final model with the best parameters from GridSearchCV.

    6. Model Evaluation:
       - Evaluate the model using confusion matrix, accuracy, precision, F1 score, and ROC-AUC score.

    7. Model Save:
       - Saves the best ANN model to the same directory the script is saved in.

    8. Visualization:
       - Visualize the training process by plotting accuracy and loss over epochs for both training and validation sets.