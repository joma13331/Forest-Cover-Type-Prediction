import logging
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from FCTPCommonTasks.FCTPFileOperations import FCTPFileOperations
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek


class FCTPFeatureEngineering:
    """
    :Class Name: FCTPFeatureEngineering
    :Description: This class is used to modify the dataframe while performing data
                  preprocessing

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        """
        :Method Name: __init__
        :Description: it initializes the logging and various variables used in the class.

        :param is_training: Whether this class has been instantiated
        """
        # Checking whether class is initialized for training or prediction
        if is_training:
            
            # Setting up the folder for training logs
            if not os.path.isdir("FCTPLogFiles/training/"):
                os.mkdir("FCTPLogFiles/training")
            # Setting up the path for the log file for this class
            self.log_path = os.path.join("FCTPLogFiles/training/", "FCTPFeatureEngineering.txt")
            
            # Variable which will tell in the logs whether log message is for training.
            self.operation = "TRAINING"
        else:

            # Setting up the folder for prediction logs
            if not os.path.isdir("FCTPLogFiles/prediction/"):
                os.mkdir("FCTPLogFiles/prediction")
            # Setting up the path for the log file for this class
            self.log_path = os.path.join("FCTPLogFiles/prediction/", "FCTPFeatureEngineering.txt")
            
            # Variable which will tell in the logs whether log message is for prediction.
            self.operation = "PREDICTION"

        # Path where the list of categorical columns are stored
        self.categorical_feat_names_path = 'FCTPRelInfo/Categorical_Features.txt'
        # Directory where models are stored
        self.models_path = "FCTPModels/"

        # File Name of the Standard Scaler object
        self.scaler_model_name = "scaler.pickle"
        # File Name of the KNNImputer object
        self.imputer_model_name = "imputer.pickle"
        # File Name of the OneHotEncoder Object
        self.ohe_model_name = "ohe.pickle"
        # File Name of the Pincipal Component Analysis Object
        self.pca_model_name = "pca.pickle"
        # Object to perform file operations
        self.file_operator = FCTPFileOperations()

        # Setting up the logging feature
        self.fctp_feature_engineering_logging = logging.getLogger("fctp_feature_engineering_log")
        self.fctp_feature_engineering_logging.setLevel(logging.INFO)
        fctp_feature_engineering_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_feature_engineering_handler.setFormatter(formatter)
        self.fctp_feature_engineering_logging.addHandler(fctp_feature_engineering_handler)

    def fctp_standard_scaling_features(self, dataframe):
        """
        :Method Name: fctp_standard_scaling_features
        :Description: This method takes in a dataframe and scales it using standard scalar
        
        :param dataframe: this is the dataframe that needs to be scaled
        
        :return: The Scaled dataset.
        :On Failure: Exception
        """
        try:
            # Checking if we are performing training or prediction
            if self.operation == 'TRAINING':
                # Creating a Standard Scaler object
                scalar = StandardScaler()
                # Creating a Pandas Datframe after performing standard scaling
                scaled_df = pd.DataFrame(scalar.fit_transform(dataframe), columns=dataframe.columns)
                
                # Logging to inform about successful scaling
                message = f"{self.operation}: The dataset has been scaled using Standard Scalar"
                self.fctp_feature_engineering_logging.info(message)

                # Saving the standard scaler object
                self.file_operator.fctp_save_model(scalar, self.models_path, self.scaler_model_name)
                return scaled_df
            else:

                # Loading the standard scaler object
                scalar = self.file_operator.fctp_load_model(os.path.join(self.models_path, self.scaler_model_name))
                
                # Creating a Pandas Datframe after performing standard scaling
                scaled_df = pd.DataFrame(scalar.transform(dataframe), columns=dataframe.columns)
                
                # Logging to inform about successful scaling
                message = f"{self.operation}: The dataset has been scaled using Standard Scalar"
                
                self.fctp_feature_engineering_logging.info(message)
                return scaled_df

        except Exception as e:
            message = f"{self.operation}: Error while trying to scale data: {str(e)}"
            self.fctp_feature_engineering_logging.error(message)
            raise e

    def fctp_handling_missing_data_mcar(self, dataframe, feature_with_missing):
        """
        :Method Name: fctp_handling_missing_data_mcar
        :Description: This method replaces the missing values if there are not greater than 75% missing using KNNImputer
        
        :param dataframe: The dataframe where null values have to be replaced
        :param feature_with_missing: The features where
        
        :return: dataframe - features with imputed values
                 dropped_features - features with more than 75% null
        :On Failure : Exception
        """
        try:

            # Checking if we are performing training or prediction
            if self.operation == 'TRAINING':
                # Creating an empty list where columns to drop will be stored
                dropped_features = []
                # Creating a KNNImputer object
                imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)

                # Looping through all the features with missing values
                for feature in feature_with_missing:

                    # Checking to find any column with more than 75% missing
                    if dataframe[feature].isna().mean > 0.75:
                        # Dropping the relevant features
                        dataframe.drop(columns=feature)
                        # Adding the column names to a list
                        dropped_features.append(feature)

                        # Logging to inform about the dropped columns
                        message = f"{self.operation}: Dropped {feature} as more than 75% values are missing"
                        self.fctp_feature_engineering_logging.info(message)

                    else:
                        # Adding a new feature which will contain information about imputed data 
                        dataframe[feature + 'nan'] = np.where(dataframe[feature].isnull(), 1, 0)

                # Imputing the nan values
                data = imputer.fit_transform(dataframe)
                # Saving the imputer model
                self.file_operator.fctp_save_model(imputer, self.models_path, self.imputer_model_name)
                # Creating a new dataframe with imputed values
                dataframe = pd.DataFrame(data=data, columns=dataframe.columns)

                # Logging to inform about successful imputation
                message = f"{self.operation}: missing values imputed using KNNImputer " \
                          f"for {list(set(feature_with_missing).symmetric_difference(set(dropped_features)))} "
                self.fctp_feature_engineering_logging.info(message)
                return dataframe, dropped_features

            else:
                # Checking to see if the imputer mode is there
                if os.path.isfile(os.path.join(self.models_path, self.imputer_model_name)):
                    # Loading the imputer model
                    imputer = self.file_operator.fctp_load_model(os.path.join(self.models_path,
                                                                            self.imputer_model_name))
                    # Performing Imputation 
                    data = imputer.transform(dataframe)
                    dataframe = pd.DataFrame(data=data, columns=dataframe.columns)
                    
                    # Logging to inform about successful imputation
                    message = f"{self.operation}: missing values imputed using KNNImputer " \
                              f"for {list(set(feature_with_missing))} "
                    self.fctp_feature_engineering_logging.info(message)
                return dataframe

        except Exception as e:
            message = f"Error while trying to handle missing data due to mcar: {str(e)}"
            self.fctp_feature_engineering_logging.error(message)
            raise e

    def fctp_pca_decomposition(self, dataframe, variance_to_be_retained):
        """
        :Method Name: fctp_pca_decomposition
        :Description: This method performs Principal Component Analysis of the dataframe passed. To be used when the
                      multi-collinear features contain information vital enough that they will be lost if removed for
                      future analysis.
        :param dataframe: The dataframe on which PCA has to be carried out to retain the information which may get lost
                          if feature removal was carried out.
        :param variance_to_be_retained: The amount of variance to be retained after PCA has been carried out.
        :return: pca_dataframe - The resultant dataframe after PCA has been carried out
        """
        try:
            if self.operation == 'TRAINING':

                pca = PCA(n_components=variance_to_be_retained, svd_solver="full")
                pca_data = pca.fit_transform(dataframe)

                feature_names = [f"Feature_{i+1}" for i in range(pca.n_components_)]
                pca_dataframe = pd.DataFrame(data=pca_data, columns=feature_names)

                message = f"{self.operation}: The Principal Component Analysis model is Trained"
                self.fctp_feature_engineering_logging.info(message)

                self.file_operator.fctp_save_model(pca, self.models_path, self.pca_model_name)
                message = f"{self.operation}: The Principal Component Analysis model is saved at {self.models_path}"
                self.fctp_feature_engineering_logging.info(message)

                message = f"{self.operation}: The Principal Component Analysis was carried out on the data and the no "\
                          f"of components in the resultant features for future pipeline are {pca.n_features_}"
                self.fctp_feature_engineering_logging.info(message)
                return pca_dataframe

            else:

                pca = self.file_operator.fctp_load_model(os.path.join(self.models_path, self.pca_model_name))

                message = f"{self.operation}: The Principal Component Analysis model is loaded from {self.models_path}"
                self.fctp_feature_engineering_logging.info(message)

                pca_data = pca.transform(dataframe)

                feature_names = [f"Feature_{i + 1}" for i in range(pca.n_components_)]
                pca_dataframe = pd.DataFrame(data=pca_data, columns=feature_names)
                pca_dataframe = pca_dataframe.round(1)

                message = f"{self.operation}: Data transformed into {pca.n_features_} using pca model"
                self.fctp_feature_engineering_logging.info(message)

                return pca_dataframe

        except Exception as e:
            message = f"{self.operation}: Error while doing pca decomposition: {str(e)}"
            self.fctp_feature_engineering_logging.error(message)
            raise e