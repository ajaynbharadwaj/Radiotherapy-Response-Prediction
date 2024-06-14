# import all used packages and functions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
# define RF_model function
def RF_model(X_train, X_test, y_train, y_test, R_STATE):

    print("-------------")
    print("Random Forest")
    print("-------------")
    # binarize train and test data with median as separator
    y_train = y_train.apply(lambda x: 1 if x > y_train.median() else 0)
    y_test = y_test.apply(lambda x: 1 if x > y_train.median() else 0)

    # Create the basic random forest model
    rf_model = RandomForestClassifier(random_state=R_STATE)
    rf_model.fit(X_train,y_train)

    # Accuracy of RF model without grid search
    print("Train accuracy w/out grid search:",rf_model.score(X_train,y_train))
    print("Test accuracy w/out grid search:",rf_model.score(X_test,y_test))

    # Define hyperparameters for grid search
    n_estimators = [50, 100, 200]
    max_features = ['sqrt', 1]
    max_depth = [2, 3, 4]
    min_samples_split = [5, 7, 10]
    min_samples_leaf = [2, 4, 6]
    bootstrap = [True, False]
    random_state = [R_STATE]

    param_grid = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap,
                  'random_state': random_state}

    # Run a grid search to find optimal parameters
    rf_grid = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    print("Grid Best Params:", rf_grid.best_params_)
    print("Train accuracy with grid search:",rf_grid.score(X_train,y_train))

    # Get predictions from the best Random Forest model according to GridSearchCV
    y_pred = rf_grid.predict(X_test)
    y_pred_proba = rf_grid.predict_proba(X_test)

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
    final_data = pd.DataFrame(score_data)
    print(final_data)
    print(f"Predicted Amount of 0: {pd.Series(y_pred).value_counts().get(0,0)} and 1: {pd.Series(y_pred).value_counts().get(1,0)}")
    print(f"Actual Amount of 0: {pd.Series(y_test).value_counts().get(0,0)} and 1: {pd.Series(y_test).value_counts().get(1,0)}")

    rf_data = {
        'Best Parameters': [rf_grid.best_params_],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'ROC-AUC': [auc(fp_rates, tp_rates)]
    }
    rf_data_df = pd.DataFrame(rf_data)
    rf_data_df.to_csv('RF_data.csv', index=False)

    # Create the ROC-AUC plot
    plt.figure()
    plt.plot(fp_rates, tp_rates, label='Classifier')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot([0, 1], [0, 1], color="r", ls="--", label='random\nclassifier')
    plt.legend()
    plt.title('ROC curve Random Forest')
    plt.tight_layout()
    plt.savefig("plots/RF_ROC.png")

    feature_names = (X_train.columns)
    best_rf = rf_grid.best_estimator_
    feature_importances = best_rf.feature_importances_

    # Get the indices of the top 10 features - not used
    indices = np.argsort(feature_importances)[::-1][:10]

    # Plot the feature importances - not used
    plt.figure(figsize=(12, 10))
    plt.title("Top 10 Feature Importances")
    plt.bar(range(len(indices)), feature_importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation = 20)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.savefig("plots/Top_10_features.png")

    print("RF Model - End")
    return 0