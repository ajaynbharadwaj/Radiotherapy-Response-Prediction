import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def SVR_model(X_train, X_test, y_train, y_test):

    print("----------------------")
    print("Support Vector Machine")
    print("----------------------")

    # hyperparameters for grid search
    param_grid = { 
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto', 1e-2, 1e-3, 1e-4],
        'epsilon': [1e-1, 1e-2, 1e-3],
        'C': [0.1, 0.4, 0.7, 1, 5, 10, 50]
    }

    # find the best parameters and then fit the model
    svr_model = SVR()
    svr_grid = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    svr_grid.fit(X_train, y_train)

    best_params = svr_grid.best_params_
    print("Best Parameters SVR:", best_params)

    y_pred_test = svr_grid.predict(X_test)
    y_pred_train = svr_grid.predict(X_train)

    # scores of train and set datasets
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    print("RMSE: Test {}, Train {}".format(rmse_test, rmse_train))
    print("R2: Test {}, Train {}".format(r2_test, r2_train))

    # plot true values vs predicted values
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)

    min_val = min(min(y_test), min(y_pred_test))
    max_val = max(max(y_test), max(y_pred_test))

    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

    plt.title('Comparison of Predicted vs True Values', fontsize=16)
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.grid(True)

    plt.savefig('plots/SVR.png', dpi=300, bbox_inches='tight')

    print("SVR Model - End")
    return 0