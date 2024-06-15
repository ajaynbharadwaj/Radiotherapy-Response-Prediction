import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def LR_model(X_train, X_test, y_train, y_test):

    print("-------------------")
    print("Logistic Regression")
    print("-------------------")

    # binarize train and test data with median as separator
    y_train = y_train.apply(lambda x: 1 if x > y_train.median() else 0)
    y_test = y_test.apply(lambda x: 1 if x > y_train.median() else 0)

    # hyperparameters for grid search
    param_grid = {
        'C': [float(x) for x in np.linspace(start=0.0001, stop=0.001, num=10)], 
        'penalty': ['l1','l2'], 
        'solver': ['liblinear']
    }
    
    # find the best parameters and then fit the model
    lr_grid = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)

    print("Best Parameters LR:", lr_grid.best_params_)

    # get the prediction from the best model found in the grid search
    y_pred = lr_grid.predict(X_test)
    y_pred_proba = lr_grid.predict_proba(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    fp_rates, tp_rates, _ = roc_curve(y_test, y_pred_proba[:,1])

    score_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        'Score': [accuracy, precision, recall, f1, auc(fp_rates,tp_rates)]
    }

    # Results of the scores and values
    final_scores = pd.DataFrame(score_data)
    print(final_scores)
    print(f"Predicted Amount of 0: {pd.Series(y_pred).value_counts().get(0,0)} and 1: {pd.Series(y_pred).value_counts().get(1,0)}")
    print(f"Actual Amount of 0: {pd.Series(y_test).value_counts().get(0,0)} and 1: {pd.Series(y_test).value_counts().get(1,0)}")

    # Create the ROC-AUC plot
    plt.figure(figsize=(8, 8))
    plt.plot(fp_rates, tp_rates, label='Classifier')
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.plot([0, 1], [0, 1], color="r", ls="--", label='random\nclassifier')
    plt.legend()
    plt.title('ROC curve Logistic Regression', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/LR_ROC.png")

    print("LR Model - End")
    return 0