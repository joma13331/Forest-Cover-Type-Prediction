import os
import json
import logging
import pandas as pd

from FCTPCommonTasks.FCTPFileOperations import FCTPFileOperations
from FCTPCommonTasks.FCTPDataLoader import FCTPDataLoader
from FCTPCommonTasks.FCTPEda import FCTPEda
from FCTPCommonTasks.FCTPFeatureEngineering import FCTPFeatureEngineering
from FCTPCommonTasks.FCTPFeatureSelection import FCTPFeatureSelection


class FCTPPredictionPipeline:
    """
    :Class Name: FCTPPredictionPipeline
    :Description: This class contains the method that will perform the prediction of the
                  data submitted by the client

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self):
        """
        :Method Name: __init__
        :Description: This constructor sets up the logging feature and paths where the models and
                      relevant information are stored
        :return: None
        """
        # Variable which will tell in the logs whether log message is for prediction.
        self.operation = 'PREDICTION'

        # Setting up the folder for prediction logs
        if not os.path.isdir("FCTPLogFiles/prediction/"):
            os.mkdir("FCTPLogFiles/prediction")
        # Setting up the path for the log file for this class
        self.log_path = os.path.join("FCTPLogFiles/prediction/", "FCTPPredictionPipeline.txt")

        # Setting up the folder for ML Models
        if not os.path.isdir("FCTPModels/FCTPMLModels/"):
            os.mkdir("FCTPModels/FCTPMLModels/")
        self.ml_model_dir = "FCTPModels/FCTPMLModels/"

        # Setting up the folder for Sklearn Models
        if not os.path.isdir("FCTPModels/"):
            os.mkdir("FCTPModels/")
        self.models_dir = "FCTPModels/"

        # Setting up the folder for Text files containing Relevant
        if not os.path.isdir("FCTPRelInfo/"):
            os.mkdir("FCTPRelInfo/")
        self.rel_info_dir = "FCTPRelInfo/"

        # File name of text file containing continuous features
        self.cont_feat_file_name = "Continuous_Features.txt"

        # Setting up the logging feature
        self.fctp_prediction_pipeline_logging = logging.getLogger("fctp_prediction_pipeline_log")
        self.fctp_prediction_pipeline_logging.setLevel(logging.INFO)
        fctp_prediction_pipeline_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_prediction_pipeline_handler.setFormatter(formatter)
        self.fctp_prediction_pipeline_logging.addHandler(fctp_prediction_pipeline_handler)
    
    
    def fctp_map_prediction_to_label(self, pred):
        """
        :Method Name: fctp_map_prediction_to_label
        :Description: This static method takes a prediction from the ML Models in this project work
                      and return the Forest Cover Type Name so that the solution of the prediction
                      dataset maybe more understandable
        
        :params pred: The prediction from the Ml Models

        :returns: The Forest Cover Type
        :On Failure: Exception
        """    

        try:
            
            if pred == 1:
                return "Spruce/Fir"
            elif pred == 2:
                return "Lodgepole Pine"
            elif pred == 3:
                return "Ponderosa Pine"
            elif pred == 4:
                return "Cottonwood/Willow"
            elif pred == 5:
                return "Aspen"
            elif pred == 6:
                return "Douglas-fir"
            else:
                return "Krummholz"
        
        except Exception as e:
            message = f"{self.operation}: There was an ERROR while performing prediction on given data: {str(e)}"
            self.fctp_prediction_pipeline_logging.error(message)
            raise e

    def fctp_predict(self):
        """
        :Method Name: fctp_predict
        :Description: This method implements the prediction pipeline which will predict on
                      the client data during deployment.
        :return: the features and their corresponding predicted labels as a json object in string format
        """
        try:

            # Initial objects setup
            message = f"{self.operation}: Start of Prediction Pipeline"
            self.fctp_prediction_pipeline_logging.info(message)

            data_loader = FCTPDataLoader(is_training=False)
            eda = FCTPEda(is_training=False)
            feature_engineer = FCTPFeatureEngineering(is_training=False)
            feature_selector = FCTPFeatureSelection(is_training=False)
            file_operator = FCTPFileOperations()

            # Loading the data
            prediction_data = data_loader.fctp_get_data()

            message = f"{self.operation}: Data to predict on obtained"
            self.fctp_prediction_pipeline_logging.info(message)

            # DATA PRE-PROCESSING

            # Removing The 'id' column
            features = feature_selector.fctp_remove_columns(prediction_data, ['id', 'ID'])
            message = f"{self.operation}: Removed the 'id' and 'ID' column"
            self.fctp_prediction_pipeline_logging.info(message)

            # Removing all columns not trained on
            with open(os.path.join(self.rel_info_dir, "columns_to_drop.txt")) as f:
                val = f.read()
            col_to_drop = val.split(",")

            # Creating an empty list so as not to raise error when no columns needs to be dropped
            if col_to_drop[0] == '':
                col_to_drop = []

            # Removing the columns
            features = feature_selector.fctp_remove_columns(features, col_to_drop)
            message = f"{self.operation}: Dropped all the irrelevant columns after feature selection"
            self.fctp_prediction_pipeline_logging.info(message)

            # Obtaining columns with 'null' values if present
            is_null_present, columns_with_null = eda.fctp_features_with_missing_values(features)
            # If null present handling it using KNNImputer.
            if is_null_present:
                features = feature_engineer.fctp_handling_missing_data_mcar(features, columns_with_null)
            message = f"{self.operation}: Checked for null values and if any were present imputed them"
            self.fctp_prediction_pipeline_logging.info(message)

            # Using PowerTransformer on features which help improving normality
            with open(os.path.join(self.rel_info_dir, self.cont_feat_file_name)) as f:
                continuous_features_str = f.read()
            continuous_features = continuous_features_str.split(',')
            features = eda.fctp_obtain_normal_features(features, continuous_features)
            message = f"{self.operation}: Converted all possible continuous columns to normal"
            self.fctp_prediction_pipeline_logging.info(message)

            # Scaling the features
            features = feature_engineer.fctp_standard_scaling_features(features)
            message = f"{self.operation}: All the features have been scaled"
            self.fctp_prediction_pipeline_logging.info(message)

            # Performing Principal Component Analysis
            features = feature_engineer.fctp_pca_decomposition(features, variance_to_be_retained=0.99)
            message = f"{self.operation}: Performed PCA and retained 99% of variance"
            self.fctp_prediction_pipeline_logging.info(message)

            message = f"{self.operation}: Data Preprocessing completed"
            self.fctp_prediction_pipeline_logging.info(message)

            # Performing clustering
            cluster = file_operator.fctp_load_model(os.path.join(self.models_dir, "cluster.pickle"))

            # Saving the cluster and the ids back to the features
            features['clusters'] = cluster.predict(features)
            features['id'] = prediction_data['id']
            features['ID'] = prediction_data['ID']

            result = []

            # Making predictions for each cluster
            for i in features["clusters"].unique():
                # Obtaining the data to prediction on for the ith cluster
                cluster_data = features[features["clusters"] == i]
                # obtaining the ids of data
                id1 = cluster_data['id']
                
                # Dropping the data not used for prediction
                cluster_data = cluster_data.drop(columns=["clusters", 'id', 'ID'])
                # Loading the model that will be used to predict on the ith cluster
                model = file_operator.fctp_load_ml_model(i)

                pred_result = list(model.predict(cluster_data))
                pred_result = [self.fctp_map_prediction_to_label(pred) for pred in pred_result]
                result.extend(list(zip(id1, pred_result)))

            res_dataframe = pd.DataFrame(data=result, columns=["id", 'Cover_Type'])
            prediction_data = prediction_data.merge(right=res_dataframe, on='id', how='outer')

            prediction_data.to_csv("prediction_result.csv", header=True, index=False)

            message = f"{self.operation}: End of EEPrediction Pipeline"
            self.fctp_prediction_pipeline_logging.info(message)

            return json.loads(prediction_data.to_json(orient="records"))

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while performing prediction on given data: {str(e)}"
            self.fctp_prediction_pipeline_logging.error(message)
            raise e