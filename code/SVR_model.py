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

    param_grid = { 
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        'gamma': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    }

    svr_model = SVR()
    svr_grid = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=6, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    svr_grid.fit(X_train, y_train)

    best_params = svr_grid.best_params_
    print("Best Parameters SVR:", best_params)

    y_pred_test = svr_grid.predict(X_test)
    y_pred_train = svr_grid.predict(X_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    print("RMSE: Test {}, Train {}".format(rmse_test, rmse_train))
    print("R2: Test {}, Train {}".format(r2_test, r2_train))

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)

    min_val = min(min(y_test), min(y_pred_test))
    max_val = max(max(y_test), max(y_pred_test))

    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

    plt.title('Comparison of Predicted vs True Values', fontsize=16)
    plt.xlabel('True Values (y_test)', fontsize=14)
    plt.ylabel('Predicted Values (y_pred_test)', fontsize=14)
    plt.grid(True)

    plt.savefig('Submission/Plots/SVR.png', dpi=300, bbox_inches='tight')

    print("SVR Model - End")
    return 0