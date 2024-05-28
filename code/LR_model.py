import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def LR_model(X_train, X_test, y_train, y_test):

    print("-------------------")
    print("Logistic Regression")
    print("-------------------")

    y_train = y_train.apply(lambda x: 1 if x > y_train.median() else 0)
    y_test = y_test.apply(lambda x: 1 if x > y_train.median() else 0)

    param_grid = {'C': [float(x) for x in np.linspace(start=0.0001, stop=0.001, num=10)], 
                  'penalty': ['l1','l2'], 
                  'solver': ['liblinear']
    }
    
    LR_grid = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    LR_grid.fit(X_train, y_train)

    best_params = LR_grid.best_params_
    print("Best Parameters LR:", best_params)

    best_LR_model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver=best_params['solver'], max_iter=1000)
    best_LR_model.fit(X_train, y_train)
    y_pred = best_LR_model.predict(X_test)


    y_pred_proba = best_LR_model.predict_proba(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    fp_rates, tp_rates, _ = roc_curve(y_test, y_pred_proba[:,1])

    score_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        'Score': [accuracy, precision, recall, f1, auc(fp_rates,tp_rates)]
    }

    # Creating DataFrame
    final_data = pd.DataFrame(score_data)
    print(final_data)
    print(f"Amount of 0: {pd.Series(y_pred).value_counts().get(0,0)} and 1: {pd.Series(y_pred).value_counts().get(1,0)}")

    print("LR Model - End")
    return 0