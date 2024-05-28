import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

def PreProcessing(R_STATE, N_FEATURES, VAR_THRESHOLD, TEST_SIZE):

    expression = pd.read_csv("data/expressionData.csv")
    sensitivity = pd.read_csv("data/radiosensitivity.csv")
    sens_cols = sensitivity.columns
    data = pd.merge(expression, sensitivity, on='cell_line_name')
    X = data[data.columns.drop(sens_cols)]
    y = pd.Series(data['SF2'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=R_STATE)
    print("Shape before processing: {}".format(X_train.shape))

    variances = X_train.var()
    cols = variances[variances >= VAR_THRESHOLD].index
    X_train_processed = X_train[cols]
    X_test_processed = X_test[cols]
    print("Shape after variance filter: {}".format(X_train_processed.shape))

    plt.figure(figsize=(12, 10))
    xmin = min(X_train.var().min(), X_train_processed.var().min())
    xmax = max(X_train.var().max(), X_train_processed.var().max())

    # First subplot: Variance before filter
    plt.subplot(2, 1, 1)
    sns.histplot(X_train.var(), kde=True, bins=20)
    plt.title('Variance of Each Feature (before filter)')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    plt.xlim(xmin, xmax)
    plt.ylim(0, 15000)

    # Second subplot: Variance after filter
    plt.subplot(2, 1, 2)
    sns.histplot(X_train_processed.var(), kde=True, bins=20, color='green')
    plt.title('Variance of Each Feature (after filter)')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    plt.xlim(xmin, xmax)
    plt.ylim(0, 15000)

    plt.tight_layout()
    plt.savefig('plots/variances_combined.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(9, 3))
    sns.boxplot(x=y_train)
    plt.title("SF2 Distribution")
    plt.xlabel('SF2 values in the train dataset')
    plt.axvline(y_train.mean(), color='r', linestyle='--', label=f'Mean: {y_train.mean():.2f}')
    plt.axvline(y_train.median(), color='g', linestyle='-', label=f'Median: {y_train.median():.2f}')
    plt.legend()
    plt.savefig('plots/SF2.png', dpi=300, bbox_inches='tight')

    mutual_info_scores = mutual_info_regression(X_train_processed, y_train, random_state=R_STATE)
    feature_scores = pd.DataFrame({'Feature': X_train_processed.columns, 'Mutual_Info_Score': mutual_info_scores})
    feature_scores_sorted = feature_scores.sort_values(by='Mutual_Info_Score', ascending=False)
    feature_scores_sorted.to_csv('data/mutual_info_sorted.csv', index=False)
    #feature_scores_sorted = pd.read_csv("data/mutual_info_sorted.csv")
    top_features = feature_scores_sorted.head(N_FEATURES)
    top_feature_names = top_features['Feature'].tolist()
    X_train_processed = X_train_processed[top_feature_names]
    X_test_processed = X_test_processed[top_feature_names]
    print("Shape after mutual info filter: {}".format(X_train_processed.shape))

    print("Final shapes: {}, {}".format(X_train_processed.shape, X_test_processed.shape))

    return X_train_processed, X_test_processed, y_train, y_test