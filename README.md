# Radiotherapy-Response-Prediction

ETH Zürich - Foundations of Data Science - FS24 Project repository of Ajay Bharadwaj, Cedric Schmucki and Julian Thüring. Radiotherapy response prediction with CCLE gene expression data.

Our code is organized into five files:

main.py, PreProcessing.py, RF_model.py, LR_model.py, and SVR_model.py.

Only main.py needs to be run, which will automatically access the functions in the other files.

Within main.py, there are four configurable variables:

• R_STATE: The seed value for reproducibility.
• N_FEATURES: The number of features to retain after feature selection.
• VAR_THRESHOLD: The threshold for variance in feature selection.
• TEST_SIZE: The proportion of the dataset to be used as the test set.

Ensure that your working directory is set to Radiotherapy-Response-Prediction. 
Please note that the expressionData.csv file was too large to include in the GitHub repository and must be manually added to the data folder.

To ensure reproducibility of the results, use R_STATE value 30052024 (the date of the project presentation). This is essential for the train-test split, mutual information regression and the random forest classifier.