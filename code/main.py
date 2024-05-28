from PreProcessing import PreProcessing
from SVR_model import SVR_model
from RF_model import RF_model
from LR_model import LR_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():

    R_STATE = 30052024
    N_FEATURES = 500
    VAR_THRESHOLD = 1
    TEST_SIZE = 0.2

    X_train, X_test, y_train, y_test = PreProcessing(R_STATE, N_FEATURES, VAR_THRESHOLD, TEST_SIZE)
    #SVR_model(X_train, X_test, y_train, y_test)
    RF_model(X_train, X_test, y_train, y_test, R_STATE)
    #LR_model(X_train, X_test, y_train, y_test)

    return 0

main()