import os
import logging
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix


class FCTPModelFinderTrain:
    """
    :Class Name: FCTPModelFinderTrain
    :Description: This class will be used to train different models and select the best one
                  amongst them.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """
    count = 0

    def __init__(self):
        """
        :Method Name: __init__
        :Description: This constructor method initializes various variables used 
                      throughout this class
        """

        # Setting up the folder for training logs
        if not os.path.isdir("FCTPLogFiles/training/"):
            os.mkdir("FCTPLogFiles/training/")
        # Setting up the path for the log file for this class
        self.log_path = "FCTPLogFiles/training/FCTPModelFinderTrain.txt"

        # Variable which will tell in the logs whether log message is for training.
        self.operation = 'TRAINING'

        # Initializing a Randon Forest Classifier object
        self.rfc = RandomForestClassifier(n_jobs=-1, verbose=0)
        # Initializing a XGBOOST Classifier object
        self.xgb = XGBClassifier(n_jobs=-1, objective='binary:logistic')
        # Initializing a Logistic Regression object
        self.logistic_regression = LogisticRegression(n_jobs=-1, max_iter=10000)
        # Initializing a SVMClassifier object
        self.svc = SVC()
        # Initializing the KFold Object
        self.kfold = KFold(shuffle=True, random_state=42)

        if FCTPModelFinderTrain.count==0:
            # Setting up the Logging feature
            self.fctp_model_finder_logging = logging.getLogger("fctp_model_finder_log")
            self.fctp_model_finder_logging.setLevel(logging.INFO)
            fctp_model_finder_handler = logging.FileHandler(self.log_path)
            formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                        datefmt='%m/%d/%Y %I:%M:%S %p')
            fctp_model_finder_handler.setFormatter(formatter)
            self.fctp_model_finder_logging.addHandler(fctp_model_finder_handler)
            FCTPModelFinderTrain.count +=1
        else:
            self.fctp_model_finder_logging = logging.getLogger("fctp_model_finder_log")

    def fctp_best_logistic_regressor(self, train_x, train_y):
        """
        :Method Name: fctp_best_logistic_regressor
        :Description: This method trains and returns the best model amongst many trained logistic regressors.

        :param train_x: Input training Data
        :param train_y: Input training labels

        :return: The best logistic regressor model
        :On failure: Exception
        """

        try:
            # creating the dictionary for parameters to be checked
            param_grid = {
                'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
            }

            # Logging about the hyperparameters of logistic regression for tuning
            message = f"{self.operation}: Using GridSearchCV to obtain the optimum parameters({param_grid.keys()})" \
                      f"  of Logistic Regressor"
            self.fctp_model_finder_logging.info(message)

            # GridSearchCV is used as there are only a few combination of parameters.
            grid = GridSearchCV(estimator=self.logistic_regression, param_grid=param_grid,
                                cv=self.kfold, n_jobs=-1,
                                scoring='accuracy',
                                verbose=0)

            # Performing the grid search on the training data
            grid.fit(train_x, train_y)

            # Storing the parameters which resulted in best model
            c = grid.best_params_['C']
            penalty = grid.best_params_['penalty']
            solver = grid.best_params_['solver']
            
            # The accuracy score
            score = grid.best_score_

            # Logging about the selected parameters
            message = f"{self.operation}: The optimum parameters of Logistic Regressor are C={c}, penalty={penalty}," \
                      f"solver={solver}  with the accuracy score of {score}"
            self.fctp_model_finder_logging.info(message)

            # Creating the best Logistic Regression object
            self.logistic_regression = LogisticRegression(C=c, penalty=penalty, solver=solver,
                                                          )
            # Training the best Logistic Regression model
            self.logistic_regression.fit(train_x, train_y)

            # Logging about the best Logistic Regression being Trained
            message = f"{self.operation}: Best Logistic  Regressor trained"
            self.fctp_model_finder_logging.info(message)

            return self.logistic_regression

        except Exception as e:
            message = f"{self.operation}: There was a problem while fitting Logistic Regressor: {str(e)}"
            self.fctp_model_finder_logging.error(message)
            raise e

    def fctp_best_svc(self, train_x, train_y):
        """
        :Method Name: fctp_best_svc
        :Description: This method trains and returns the best model amongst many trained SVC.

        :param train_x: Input training Data
        :param train_y: Input training labels
        :return: The best SVC model
        :On failure: Exception
        """

        try:
            # creating the dictionary for parameters to be checked
            param_grid = {
                'C': [ 0.3, 1, 3, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4, 5],
                'gamma': ['scale', 'auto']
            }

            # Logging about the hyperparameters of SVC for tuning
            message = f"{self.operation}: Using GridSearchCV to obtain the optimum parameters({param_grid.keys()})" \
                      f" of SVC"
            self.fctp_model_finder_logging.info(message)

            # GridSearchCV is used as there are only a few combination of parameters.
            grid = GridSearchCV(estimator=self.svc, param_grid=param_grid,
                                cv=self.kfold, n_jobs=-1,
                                scoring='accuracy',
                                verbose=0)

            # Performing the grid search on the training data
            grid.fit(train_x, train_y)

            # Storing the parameters which resulted in best model
            kernel = grid.best_params_['kernel']
            gamma = grid.best_params_['gamma']
            c = grid.best_params_['C']
            degree = grid.best_params_['degree']

            # The accuracy score
            score = grid.best_score_

            # Logging about the selected parameters
            message = f"{self.operation}: The optimum parameters of SVC are kernel={kernel}, gamma={gamma}, C={c}," \
                      f" degree ={degree} with the accuracy score of {score}"
            self.fctp_model_finder_logging.info(message)

            # Creating the best SVC object
            self.svc = SVC(kernel=kernel, gamma=gamma, C=c, degree=degree)
            # Training the best SVC model 
            self.svc.fit(train_x, train_y)

            # Logging about the best SVC being Trained
            message = f"{self.operation}: Best SVC trained"
            self.fctp_model_finder_logging.info(message)

            return self.svc

        except Exception as e:
            message = f"{self.operation}: There was a problem while fitting SVC: {str(e)}"
            self.fctp_model_finder_logging.error(message)
            raise e

    def fctp_best_random_forest(self, train_x, train_y):
        """
        :Method Name: fctp_best_random_forest
        :Description: This method trains and returns the best model amongst many trained random forest classifier.

        :param train_x: Input training Data
        :param train_y: Input training labels
        :return: The best random forest classifier model
        :On failure: Exception"""

        try:
            # creating the dictionary for parameters to be checked
            param_grid = {
                'n_estimators': [100, 130, 150, 300],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 3, 4, 5, 6],
                'max_features': ['auto', 'sqrt', 'log2'],
                'ccp_alpha': np.arange(0, 0.01, 0.001)
            }
            
            # Logging about the hyperparameters of Random Forest Classifier for tuning
            message = f"{self.operation}: Using RandomSearchCV to obtain the optimum parameters({param_grid.keys()})" \
                      f" of random forest classifier "
            self.fctp_model_finder_logging.info(message)

            # RandomSearchCV is used as there are a large number combination of parameters.
            grid = RandomizedSearchCV(estimator=self.rfc, param_distributions=param_grid, n_iter=500,
                                      cv=self.kfold, n_jobs=-1,
                                      scoring='accuracy',
                                      verbose=0)
            # Performing the grid search on the training data
            grid.fit(train_x, train_y)

            # Storing the parameters which resulted in best model
            n_estimators = grid.best_params_['n_estimators']
            criterion = grid.best_params_['criterion']
            min_samples_split = grid.best_params_['min_samples_split']
            max_features = grid.best_params_['max_features']
            ccp_alpha = grid.best_params_['ccp_alpha']

            # The accuracy score
            score = grid.best_score_
            # Logging about the selected parameters
            message = f"{self.operation}: The optimum parameters of random forrest classifier are " \
                      f"n_estimators={n_estimators}, criterion={criterion}, min_samples_split={min_samples_split}," \
                      f" max_features ={max_features}, ccp_alpha={ccp_alpha} with the accuracy of {score}"
            self.fctp_model_finder_logging.info(message)
            
            # Creating the best Random Forest Classifier object
            self.rfc = RandomForestClassifier(n_jobs=-1, verbose=0,
                                              n_estimators=n_estimators, criterion=criterion,
                                              min_samples_split=min_samples_split,
                                              max_features=max_features, ccp_alpha=ccp_alpha
                                              )
            # Training the best Random Forest Classifier model
            self.rfc.fit(train_x, train_y)

            # Logging about the best Random Forest Classifier being Trained
            message = f"{self.operation}: Best random forest classifier trained"
            self.fctp_model_finder_logging.info(message)

            return self.rfc

        except Exception as e:
            message = f"{self.operation}: There was a problem while fitting Random Forest classifiers: {str(e)}"
            self.fctp_model_finder_logging.error(message)
            raise e

    def fctp_best_xgb_classifier(self, train_x, train_y):
        """
        :Method Name: fctp_best_xgb_classifier
        :Description: This method trains and returns the best model amongst many trained xgb classifiers.

        :param train_x: Input training Data
        :param train_y: Input training labels
        :return: The best xgb classifier model
        :On failure: Exception
        """

        try:
            # creating the dictionary for parameters to be checked
            param_grid = {
                'learning_rate': [0.01, 0.03, 0.1, 0.3, 1],
                'colsample_bytree': [.1, .2, .3, .4, .5, .6, .7, .8],
                'max_depth': [10, 15, 20],
                'n_estimators': [300, 1000, 3000],
                "verbosity": [1]
            }

            # Logging about the hyperparameters of XGB Classifier for tuning
            message = f"{self.operation}: Using RandomSearchCV to obtain the optimum parameters({param_grid.keys()}) " \
                      f"of xgb classifier"
            self.fctp_model_finder_logging.info(message)

            # RandomSearchCV is used as there are a large number combination of parameters.
            grid = RandomizedSearchCV(estimator=self.xgb, param_distributions=param_grid, n_iter=100,
                                      cv=5, n_jobs=-1,
                                      scoring='accuracy', verbose=1)

            # Performing the grid search on the training data
            grid.fit(train_x, train_y)

            # Storing the parameters which resulted in best model
            learning_rate = grid.best_params_['learning_rate']
            colsample_bytree = grid.best_params_['colsample_bytree']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # The accuracy score
            score = grid.best_score_

            # Logging about the selected parameters
            message = f"{self.operation}: The optimum parameters of xgb-classifier are learning_rate={learning_rate}, "\
                      f"max_depth={max_depth}, colsample_bytree={colsample_bytree}, n_estimators ={n_estimators} " \
                      f"with the accuracy score of {score}"
            self.fctp_model_finder_logging.info(message)

            # Creating the best XGBClassifier object
            self.xgb = XGBClassifier(n_jobs=-1, verbose=0, learning_rate=learning_rate,
                                     colsample_bytree=colsample_bytree,
                                     max_depth=max_depth, n_estimators=n_estimators)

            # Training the best XGBClassifier model
            self.xgb.fit(train_x, train_y)

            # Logging about the best XGBClassifier being Trained
            message = f"{self.operation}: Best xgb classifier trained"
            self.fctp_model_finder_logging.info(message)

            return self.xgb

        except Exception as e:
            message = f"{self.operation}: There was a problem while fitting XGB Classifier: {str(e)}"
            self.fctp_model_finder_logging.error(message)
            raise e
            
    def fctp_best_model_from_accuracy(self, accuracy_scores):
        """
        :Method Name: fctp_best_model_from_accuracy
        :Description: This method takes in a dictionary with model name as keys and accuracy score as values,
                      it then returns the best model based on highest accuracy score.

        :param accuracy_scores: The dictionary of all accuracy scores
        :return: The best sklearn model for the given dataset
        :On Failure: Exception
        """
        try:
            # List of model acronym 
            keys = list(accuracy_scores.keys())
            # List of accuracy 
            values = list(accuracy_scores.values())
            # Index of the score with maximum value
            ind = values.index(max(values))

            # Checking which key has the maximum score
            if keys[ind] == "logistic":
                # Logging about Logistic Regression as Best Model
                message = f"{self.operation}: The best model is logistic regressor with accuracy of {values[ind]}"
                self.fctp_model_finder_logging.info(message)
                return keys[ind], self.logistic_regression

            elif keys[ind] == "svc":
                # Logging about SVC as Best Model
                message = f"{self.operation}: The best model is svc with accuracy of {values[ind]}"
                self.fctp_model_finder_logging.info(message)
                return keys[ind], self.svc

            elif keys[ind] == "rfc":
                # Logging about Random Forest Classifier as Best Model
                message = f"{self.operation}: The best model is random forest classifier with accuracy of {values[ind]}"
                self.fctp_model_finder_logging.info(message)
                return keys[ind], self.rfc

            else:
                # Logging about XGB Classifier as Best Model
                message = f"{self.operation}: The best model is xgb classifier with accuracy of {values[ind]}"
                self.fctp_model_finder_logging.info(message)
                return keys[ind], self.xgb

        except Exception as e:
            message = f"{self.operation}: There was a problem while obtaining best model from accuracy " \
                      f"dictionary: {str(e)}"
            self.fctp_model_finder_logging.error(message)
            raise e

    def fctp_best_model(self, train_x, train_y, test_x, test_y):
        """
        :Method Name: fctp_best_model
        :Description: This method is used to select the best model from all the best model from all categories.

        :param train_x: the training features
        :param train_y: the training labels
        :param test_x: the test features
        :param test_y: the test labels
        :return: The best sklearn model for the given dataset
        :On Failure: Exception
        """

        try:
            
            # Logging about the start of Best Model Search
            message = f"{self.operation}: Search for best model started"
            self.fctp_model_finder_logging.info(message)

            # Dictionary of Evaluation Scores
            roc_auc_scores = {}
            f1_scores = {}
            accuracy_scores = {}
            confusion_matrices = {}
            
            # Logging about the start of search of the Best Logistic Regression Model
            message = f"{self.operation}: Search for best logistic regressor model started"
            self.fctp_model_finder_logging.info(message)

            self.logistic_regression = self.fctp_best_logistic_regressor(train_x, train_y)
            y_pred = self.logistic_regression.predict(test_x)

            # Evaluation Metrics For Best Logistic Regression Model
            # roc_auc_scores["logistic"] = roc_auc_score(y_true=test_y, y_score=y_pred)
            # f1_scores["logistic"] = f1_score(y_true=test_y, y_pred=y_pred)
            accuracy_scores["logistic"] = accuracy_score(y_true=test_y, y_pred=y_pred)
            # confusion_matrices["logistic"] = confusion_matrix(y_true=test_y, y_pred=y_pred)

            # Logging about the end of search of the Best Logistic Regression Model
            message = f"{self.operation}: Search for best logistic regressor model ended"
            self.fctp_model_finder_logging.info(message)

            # Logging about the start of search of the Best SVC Model
            message = f"{self.operation}: Search for best svc model started"
            self.fctp_model_finder_logging.info(message)

            self.svc = self.fctp_best_svc(train_x, train_y)
            y_pred = self.svc.predict(test_x)
            # Evaluation Metrics For Best SVC Model
            # roc_auc_scores["svc"] = roc_auc_score(y_true=test_y, y_score=y_pred)
            # f1_scores["svc"] = f1_score(y_true=test_y, y_pred=y_pred)
            accuracy_scores["svc"] = accuracy_score(y_true=test_y, y_pred=y_pred)
            # confusion_matrices["svc"] = confusion_matrix(y_true=test_y, y_pred=y_pred)

            # Logging about the end of search of the Best SVC Model
            message = f"{self.operation}: Search for best svc model ended"
            self.fctp_model_finder_logging.info(message)

            # Logging about the start of search of the Best Random Forest Classifier Model
            message = f"{self.operation}: Search for best random forest classifier model started"
            self.fctp_model_finder_logging.info(message)

            self.rfc = self.fctp_best_random_forest(train_x, train_y)
            y_pred = self.rfc.predict(test_x)
            # Evaluation Metrics For Best Random Forest Classifier Model
            # roc_auc_scores["rfc"] = roc_auc_score(y_true=test_y, y_score=y_pred)
            # f1_scores["rfc"] = f1_score(y_true=test_y, y_pred=y_pred)
            accuracy_scores["rfc"] = accuracy_score(y_true=test_y, y_pred=y_pred)
            # confusion_matrices["rfc"] = confusion_matrix(y_true=test_y, y_pred=y_pred)

            # Logging about the end of search of the Best Random Forest Classifier Model
            message = f"{self.operation}: Search for best random forest classifier model ended"
            self.fctp_model_finder_logging.info(message)

            # Logging about the start of search of the Best XGB Classifier Model
            message = f"{self.operation}: Search for best xgb classifier model started"
            self.fctp_model_finder_logging.info(message)

            self.xgb = self.fctp_best_xgb_classifier(train_x, train_y)
            y_pred = self.xgb.predict(test_x)
            # Evaluation Metrics For Best XGB Classifier Model
            # roc_auc_scores["xgb"] = roc_auc_score(y_true=test_y, y_score=y_pred)
            # f1_scores["xgb"] = f1_score(y_true=test_y, y_pred=y_pred)
            accuracy_scores["xgb"] = accuracy_score(y_true=test_y, y_pred=y_pred)
            # confusion_matrices["xgb"] = confusion_matrix(y_true=test_y, y_pred=y_pred)

            # Logging about the end of search of the Best XGB Classifier Model
            message = f"{self.operation}: Search for best xgb classifier model ended"
            self.fctp_model_finder_logging.info(message)

            print(f"roc_auc scores: {roc_auc_scores}")
            print(f"f1 scores:{f1_scores}")
            print(f"accuracy scores:{accuracy_scores}")
            print(f"confusion matrices: {confusion_matrices}")

            return self.fctp_best_model_from_accuracy(accuracy_scores)

        except Exception as e:
            message = f"{self.operation}: There was a problem while obtaining best model : {str(e)}"
            self.fctp_model_finder_logging.error(message)
            raise e
    