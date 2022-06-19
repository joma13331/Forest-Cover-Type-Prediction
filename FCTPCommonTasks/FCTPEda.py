import os
import logging
import numpy as np
from scipy.stats import normaltest
from sklearn.preprocessing import PowerTransformer
from FCTPCommonTasks.FCTPFileOperations import FCTPFileOperations

class FCTPEda:
    """
    :Class Name: FCTPEda
    :Description: This class is used to explore the data given by the client and come
                  to some conclusion about the data.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        """
        :Method Name: __init__
        :Description: This method is Constructor for class FCTPEda.
                      Initializes variables for logging and variables that
                      will be used throughout the class
        :param is_training: whether this class is instantiated for training purpose.
        """

        # Checking if the object is initialized for training or prediction
        if is_training:

            # Setting up the folder for training logs
            if not os.path.isdir("FCTPLogFiles/training"):
                os.mkdir("FCTPLogFiles/training")
            # Setting up the path for the log file for this class
            self.log_path = os.path.join("FCTPLogFiles/training", "FCTPEda.txt")
            
            # Variable which will tell in the logs whether log message is for training.
            self.operation = "TRAINING"

        else:
            # Setting up the folder for prediction logs
            if not os.path.isdir("FCTPLogFiles/prediction"):
                os.mkdir("FCTPLogFiles/prediction")
            # Setting up the path for the log file for this class    
            self.log_path = os.path.join("FCTPLogFiles/prediction", "FCTPEda.txt")

            # Variable which will tell in the logs whether log message is for prediction.
            self.operation = "PREDICTION"

        # Path where names of all numerical features will be stored
        self.numerical_feat_names_path = 'FCTPRelInfo/Numerical_Features.txt'
        # Path where names of all categorical features will be stored
        self.categorical_feat_names_path = 'FCTPRelInfo/Categorical_Features.txt'
        # Path where names of all continuous features will be stored
        self.cont_feat_names_path = 'FCTPRelInfo/Continuous_Features.txt'
        # Path where names of all discrete features will be stored
        self.discrete_feat_names_path = 'FCTPRelInfo/Discrete_Features.txt'
        # Path where names of all normal features will be stored
        self.normal_feature_path = 'FCTPRelInfo/Normal_Features.txt'
        # Path where names of all the features which are power transformed are stored
        self.power_transformed_feature = 'FCTPRelInfo/Power_Transformed_Features.txt'
        # Path where names of all the features which are log transformed are stored
        self.log_transformed_feature = 'FCTPRelInfo/Log_Features.txt'
        # Directory where models will be saved
        self.model_dir = "FCTPModels/"
        # Name of power transformer model
        self.power_transformer_model_name = "power_transformer.pickle"
        # object of File operator
        self.file_operator = FCTPFileOperations()

        # Setting up the logging feature
        self.fctp_eda_logging = logging.getLogger("fctp_eda_log")
        self.fctp_eda_logging.setLevel(logging.INFO)
        fctp_eda_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_eda_handler.setFormatter(formatter)
        self.fctp_eda_logging.addHandler(fctp_eda_handler)

    def fctp_feature_label_split(self, dataframe, label_col_names):
        """
        :Method Name: fctp_feature_label_split
        :Description: This method splits the features and labels from the validated
                      dataset and it returns them

        :param dataframe: The pandas dataframe to obtain features and labels from
        :param label_col_names: the name of label columns
        :return: features - a pandas dataframe composed of all the features
                 labels - a dataseries representing the output
        :On Failure: Exception
        """

        try:
            # Obtaining a separate dataframe for the features
            features = dataframe.drop(columns=label_col_names)
            # Obtaining a separate dataframe for the label
            labels = dataframe[label_col_names]

            # Logging The succesful separation of labels and features
            message = f"{self.operation}: The features and labels have been obtained from the dataset"
            self.fctp_eda_logging.info(message)

            return features, labels

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while obtaining feature and labels: {str(e)}"
            self.fctp_eda_logging.error(message)
            raise e
    
    def fctp_features_with_missing_values(self, dataframe):
        """
        :Method Name: fctp_features_with_missing_values
        :Description: This method finds out whether there are missing values in the
                      validated data and returns a list of feature names with missing
                      values

        :param dataframe: the Dataframe in which features with missing values are
                          required to be found
        :return: missing_val_flag - whether the dataframe has missing values or not
                 features_with_missing - If missing values are present then list of
                 columns with missing values otherwise an empty list
        :On Failure: Exception
        """
        try:

            # Obtaining the features will null nan values
            features_with_missing = [feature for feature in dataframe.columns if dataframe[feature].isna().sum() > 0]
            
            missing_val_flag = False
            # Setting the Flag if there are missing values in the dataset.
            if len(features_with_missing) > 0:
                missing_val_flag = True

            # Logging to inform about missing values in the dataset.
            message = f"{self.operation}: There are {len(features_with_missing)} features with missing values"
            self.fctp_eda_logging.info(message)

            return missing_val_flag, features_with_missing

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while obtaining feature and labels: {str(e)}"
            self.fctp_eda_logging.error(message)
            raise e
    
    def fctp_numerical_and_categorical_columns(self, dataframe):
        """
        :Method Name: fctp_numerical_and_categorical_columns
        :Description: This method return lists of numerical and categorical features in a dataframe

        :param dataframe:The dataframe from which the column name of numerical and categorical features have to be
                          obtained

        :return: numerical_features - List of all the numerical columns in the dataframe
                 categorical_features - List of all the categorical columns in the dataframe
        :On Failure: Exception
        """
        try:

            # Checking if we are performing training or prediction
            if self.operation == 'TRAINING':
                
                # Getting all the numerical features in the datasets
                numerical_features = [feature for feature in dataframe.columns if dataframe[feature].dtypes != 'O']
                # Obtaining the string of all numerical columns separated by a comma
                numerical_features_str = ",".join(numerical_features)

                # Writing the string of numerical features to a txt file
                with open(self.numerical_feat_names_path, 'w') as f:
                    f.write(numerical_features_str)

                # Logging about the numerical features
                message = f'{self.operation}: {numerical_features_str} are the Numerical features'
                self.fctp_eda_logging.info(message)

                # Getting all the categorical features in the datasets
                categorical_features = [feature for feature in dataframe.columns if dataframe[feature].dtypes == 'O']
                
                # Obtaining the string of all categorical columns separated by a comma
                categorical_features_str = ",".join(categorical_features)

                # Writing the string of categorical features to a txt file
                with open(self.categorical_feat_names_path, 'w') as f:
                    f.write(categorical_features_str)

                # Writing the string of categorical features to a txt file
                message = f'{self.operation}: {categorical_features_str} are the Categorical features'
                self.fctp_eda_logging.info(message)
            else:
                
                # Obtaining the list of numerical features from external text file
                with open(self.numerical_feat_names_path, 'r') as f:
                    numerical_features_str = f.read()

                # Spliiting the string of numerical features into a list
                numerical_features = numerical_features_str.split(',')

                # Logging about the numerical features
                message = f'{self.operation}: {numerical_features_str} are the Numerical features'
                self.fctp_eda_logging.info(message)

                # Obtaining the list of categorical features from external text file
                with open(self.categorical_feat_names_path, 'r') as f:
                    categorical_features_str = f.read()

                # Spliiting the string of categorical features into a list
                categorical_features = categorical_features_str.split(',')

                # Logging about the categorical features
                message = f'{self.operation}: {categorical_features_str} are the Categorical features'
                self.fctp_eda_logging.info(message)

            return numerical_features, categorical_features

        except Exception as e:
            message = f'{self.operation}: Error in obtaining the Numerical and Categorical ' \
                      f'features from the data: {str(e)}'
            self.fctp_eda_logging.error(message)

    def fctp_continuous_discrete_variables(self, dataframe, num_col):
        """
        :Method Name: fctp_continuous_discrete_variables
        :Description: This method return lists of continuous and discrete features in a dataframe

        :param dataframe: The dataframe from which the column name of continuous and discrete features have to be
                          obtained
        :param num_col: List of all the numerical columns in the dataframe

        :return: cont_feat - list of continuous features in the given dataframe
                 discrete_feat - list of discrete features in the given dataframe
        :On Failure: Exception
        """
        try:

            # Checking if we are performing training or prediction
            if self.operation == 'TRAINING':
                
                # Obtaining all the continuous variables in the features
                cont_feat = [feature for feature in num_col if len(dataframe[feature].unique()) >= 25]
                # Converting the list of continuous features to a string
                cont_feat_str = ",".join(cont_feat)

                # Writing the list of continuous variables to a txt file
                with open(self.cont_feat_names_path, 'w') as f:
                    f.write(cont_feat_str)

                # Logging about the Continous features
                message = f'{self.operation}: {cont_feat_str} are the continuous features'
                self.fctp_eda_logging.info(message)

                # Obtaining all the discrete variables in the features
                discrete_feat = [feature for feature in num_col if len(dataframe[feature].unique()) < 25]
                
                # Converting the list of discrete features to a string
                discrete_feat_str = ",".join(discrete_feat)
                
                # Writing the list of discrete variables to a txt file
                with open(self.discrete_feat_names_path, 'w') as f:
                    f.write(discrete_feat_str)

                # Logging about the discrete features
                message = f'{self.operation}: {discrete_feat_str} are the Discrete features'
                self.fctp_eda_logging.info(message)

            else:
                
                # Obtaining the string of continuous variables from the text file
                with open(self.cont_feat_names_path, 'r') as f:
                    cont_feat_str = f.read()
                
                # Obtaining the list of continuous variables
                cont_feat = cont_feat_str.split(',')

                # Logging about the continuous variables
                message = f'{self.operation}: {cont_feat_str} are the continuous features'
                self.fctp_eda_logging.info(message)

                # Obtaining the string of discrete variables from the text file
                with open(self.discrete_feat_names_path, 'r') as f:
                    discrete_feat_str = f.read()
                
                # Obtaining the list of discrete variables
                discrete_feat = discrete_feat_str.split(',')

                # Logging about the discrete variables
                message = f'{self.operation}: {discrete_feat_str} are the Discrete features'
                self.fctp_eda_logging.info(message)

            return cont_feat, discrete_feat

        except Exception as e:
            message = f'{self.operation}: Error in obtaining the Continuous and Discrete ' \
                      f'features from the data: {str(e)}'
            self.fctp_eda_logging.error(message)

    def fctp_normal_not_normal_distributed_features(self, dataframe, cont_columns):
        """
        :Method Name: fctp_normal_not_normal_distributed_features
        :Description: This method checks whether the continuous feature is normal 
                      or not.

        :param dataframe: the dataframe which needs to be checked for normal and not normal features.
        :param cont_columns: the list of continuous columns

        :return: normal_features - list of normal features
                 not_normal_features - list of features which are not normal
        :On Failure: Exception
        """
        try:

            normal_features = []
            not_normal_features = []

            # Checking all the continuous features 
            for feature in cont_columns:
                # Checking whether the feature is normal or not
                if normaltest(dataframe[feature].values)[1] >= 0.05:
                    normal_features.append(feature)
                else:
                    not_normal_features.append(feature)
            
            # Logging about the Normal Features
            message = f'{self.operation}: {normal_features} are originally normal'
            self.fctp_eda_logging.info(message)

            # Storing all the normal features in a string
            normal_features_str = ','.join(normal_features)

            # Storing the string of normal features in a text file
            with open(self.normal_feature_path, 'w') as f:
                f.write(normal_features_str)

            return normal_features, not_normal_features

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while obtaining normal " \
                      f"features and features which are not normal: {str(e)}"
            self.fctp_eda_logging.error(message)
            raise e

    def fctp_obtain_normal_features(self, dataframe, cont_columns):
        """
        :Method Name: fctp_obtain_normal_features
        :Description: This method uses the sklearn power transform model to 
                      convert the the features to normal if possible
        :param dataframe: The dataframe which needs to convert its columns to normal if possible.
        :param cont_columns: the features which are continuous in nature
        :return: Dataframe with continous columns converted to normal if possible
        """
        try:
            # Checking if we are performing training or prediction
            if self.operation == 'TRAINING':
                
                # Obtaining the features which were not normal
                normal_features, not_normal_features = self.fctp_normal_not_normal_distributed_features(dataframe,
                                                                                                       cont_columns)

                # Creating the list consisting of features that can be power transformed                                                                                      
                feature_power_transformed = []
                
                # Looping through all the features that are not normal
                for feature in not_normal_features:

                    # Creating a power Transformer object
                    power_transformer_temp = PowerTransformer()
                    # Transforming the not normal features
                    transformed_data = power_transformer_temp.fit_transform(np.array(dataframe[feature]).reshape(-1, 1))

                    # Checking whether the feature has been converted to more normal 
                    if normaltest(transformed_data)[0] < normaltest(dataframe[feature])[0]:
                        feature_power_transformed.append(feature)

                # Saving all the feature that can be transformed to more normal
                feature_power_transformed_str = ",".join(feature_power_transformed)

                # Writing all the power transformed variables
                with open(self.power_transformed_feature, 'w') as f:
                    f.write(feature_power_transformed_str)

                # Creating a new PowerTransformer object which will tranform all the relevant
                # variables
                power_transformer = PowerTransformer()
                # Transforming the variables to a more normal one
                dataframe[feature_power_transformed] = power_transformer.fit_transform(
                    dataframe[feature_power_transformed])
                # Saving the power transformer model    
                self.file_operator.fctp_save_model(power_transformer, self.model_dir, self.power_transformer_model_name)

            else:
                # Accessing all the variables that need to be power transformed
                with open(self.power_transformed_feature, 'r') as f:
                    feature_power_transformed_str = f.read()

                # Getting list of all the variables that need to be Poer Transformed    
                feature_power_transformed = feature_power_transformed_str.split(",")

                # Loading the power transformer model 
                power_transformer = self.file_operator.fctp_load_model(os.path.join(self.model_dir,
                                                                                   self.power_transformer_model_name))
                # Transforming all the relevant variables to a more normal one
                dataframe[feature_power_transformed] = power_transformer.transform(dataframe[feature_power_transformed])

            return dataframe

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while converting all possible columns to normal " \
                      f"features : {str(e)}"
            self.fctp_eda_logging.error(message)
            raise e