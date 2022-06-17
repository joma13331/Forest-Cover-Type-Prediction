import os
import pandas as pd
import logging
from sklearn.feature_selection import mutual_info_classif

class FCTPFeatureSelection:
    """
    :Class Name: FCTPFeatureSelectionTrain
    :Description: This class is used to select the features for both training as well make the same selection
                  for training.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        """
        :Method Name: __init__
        :Description: This method sets up the path variables and initializes variables for logging.

        :params is_training: Whether the object is initialized for training or not
        """

        # Checking if the object is initialized for training or prediction
        if is_training:

            # Variable which will tell in the logs whether log message is for training.
            self.operation = 'TRAINING'
            # Setting up the folder for training logs
            if not os.path.isdir("FCTPLogFiles/training/"):
                os.mkdir("FCTPLogFiles/training/")
            # Setting up the path for the log file for this class
            self.log_path = os.path.join("FCTPLogFiles/training/", "FCTPFeatureSelection.txt")
        else:
            # Variable which will tell in the logs whether log message is for prediction.
            self.operation = 'PREDICTION'
            # Setting up the folder for prediction logs
            if not os.path.isdir("FCTPLogFiles/prediction/"):
                os.mkdir("FCTPLogFiles/prediction/")
            # Setting up the path for the log file for this class
            self.log_path = os.path.join("FCTPLogFiles/prediction/", "FCTPFeatureSelection.txt")

        # Setting up the logging feature
        self.fctp_feature_selection_logging = logging.getLogger("fctp_feature_selection_log")
        self.fctp_feature_selection_logging.setLevel(logging.INFO)
        fctp_feature_selection_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_feature_selection_handler.setFormatter(formatter)
        self.fctp_feature_selection_logging.addHandler(fctp_feature_selection_handler)

    def fctp_remove_columns(self, dataframe, columns):
        """
        :Method Name: fctp_remove_columns
        :Description: This method is used to delete columns from a pandas dataframe.

        :param dataframe: The pandas dataframe from which the columns have to be
                          removed.
        :param columns: The columns that have to be removed.

        :return: A pandas dataframe with the columns removed.
        :On Failure: Exception
        """
        try:
            # Dropping the columns from the dataframe
            dataframe = dataframe.drop(columns=columns)
            # Logging to inform that the columns were dropped
            message = f"{self.operation}: The following columns were dropped: {columns}"
            self.fctp_feature_selection_logging.info(message)
            return dataframe

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while deleting columns: {str(e)}"
            self.fctp_feature_selection_logging.error(message)
            raise e

    def fctp_col_with_high_correlation(self, dataframe, threshold=0.8):
        """
        :Method Name: fctp_col_with_high_correlation
        :Description: This method finds out those features which can be removed to remove multi-collinearity.

        :param dataframe: The pandas dataframe to check for features with multi-collinearity
        :param threshold: The threshold above which features are taken to be collinear
        
        :return: A list of features that can be dropped to remove multi-collinearity
        :On Failure: Exception
        """
        try:
            
            col_corr = set()  # Set of all the names of correlated columns
            # Obtaining the Corelation matrix
            corr_matrix = dataframe.corr()

            # Looping through all the columns in the dataframe
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                        colname = corr_matrix.columns[i]  # getting the name of column
                        col_corr.add(colname)

            # Logging to inform about variables with high correlation
            message = f"{self.operation}: The following columns have high correlation with other " \
                      f"columns {str(col_corr)}"
            self.fctp_feature_selection_logging.info(message)
            return list(col_corr)

        except Exception as e:
            message = f"There was an ERROR while detecting collinear columns in features: {str(e)}"
            self.fctp_feature_selection_logging.error(message)
            raise e

    def fctp_feature_not_important(self, features, label, threshold=0.1):
        """
        :Method Name: fctp_feature_not_important
        :Description: This method determined those features which are not important to determine the output label

        :param features: The input features of the dataset provided by the client
        :param label: The output label being considered for determining feature to drop
        :param threshold: the value below which if columns have value they can be removed
        
        :return: A list of features that can be dropped as they have no impact on output label
        :On Failure: Exception
        """
        try:

            # Check the dependecy for input features to that of the labels
            mutual_info = mutual_info_classif(features, label)
            # Form a series of the result
            feature_imp = pd.Series(mutual_info, index=features.columns)
            # Obtain a list of features not important wrt a threshold
            not_imp = list(feature_imp[feature_imp < threshold].index)
            
            # Logging to inform about features not important
            message = f"{self.operation}: The features which have no or very impact on the output are {not_imp}"
            self.fctp_feature_selection_logging.info(message)

            return not_imp

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while detecting columns in features with no impact on " \
                      f"output: {str(e)} "
            self.fctp_feature_selection_logging.error(message)
            raise e

    def fctp_features_with_zero_std(self, dataframe):
        """
        :Method Name: fctp_features_with_zero_std
        :Description: This method checks whether any of the columns of the passed
                      dataframe has all values as equal and returns a list of all such
                      columns

        :param dataframe: The pandas dataframe to check for columns with all values as same
        
        :return: list of columns with zero std
        :On Failure: Exception
        """
        try:
            # Creating an empty list to store columns with only a single value
            columns_zero_std = []
            
            # Looping through all the columns of the input features
            for feature in dataframe.columns:
                
                # Checking whether standard deviation is 0
                if dataframe[feature].std() == 0:
                    columns_zero_std.append(feature)

            # Logging to inform about the columns with zero STD
            message = f"{self.operation}: the features with all values as equal are {columns_zero_std}"
            self.fctp_feature_selection_logging.info(message)

            return columns_zero_std

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while detecting columns all values as equal: {str(e)}"
            self.fctp_feature_selection_logging.error(message)
            raise e