import os
import pickle
import logging


class FCTPFileOperations:
    """
    :Class Name: FCTPFileOperations
    :Description: This class shall be used to save the model after training
    and load the saved model for prediction.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0

    """

    def __init__(self):
        """
        :Method Name: __init__
        :Description: Initializes the logging feature
        """

        # Creating the folder for logs if not already created
        if not os.path.isdir("FCTPLogFiles/"):
            os.mkdir("FCTPLogFiles/")
        # Setting up the Log File for the class FCTPFileOperations
        self.log_path = os.path.join("FCTPLogFiles/", "FCTPFileOperation.txt")

        # Setting up the logging feature
        self.fctp_file_operations_logging = logging.getLogger("fctp_file_operations_log")
        self.fctp_file_operations_logging.setLevel(logging.INFO)
        fctp_file_operation_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_file_operation_handler.setFormatter(formatter)
        self.fctp_file_operations_logging.addHandler(fctp_file_operation_handler)

    def fctp_save_model(self, model, model_dir, model_name):
        """
        :Method Name: fctp_save_model
        :Description: This method saves the passed model to the given directory

        :param model: The sklearn model to save.
        :param model_dir: The folder/directory where model need to be stored
        :param model_name: the name of the model
        :return: None
        :On Failure: Exception
        """
        try:
            # path where the model needs to be saved
            path = os.path.join(model_dir, model_name)
            
            # If the model directory is not present then create it
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

            # Saving the model to the path 
            with open(path, 'wb') as f:
                pickle.dump(model, f)

            # Logging to inform that the model has been saved
            message = f"{model_name} has been save in {model_dir}"
            self.fctp_file_operations_logging.info(message)

        except Exception as e:
            message = f"Error while save {model_name} in {model_dir}: {str(e)}"
            self.fctp_file_operations_logging.error(message)
            raise e

    def fctp_load_model(self, model_path):
        """
        :Method Name: fctp_load_model
        :Description: This method is used to obtain the model stored at the given file path.

        :param model_path: The path where model is stored.
        :return: The model stored at the passed path.
        :On Failure: Exception
        """

        try:

            # obtaining the variable to store the information about the model
            f = open(model_path, 'rb')

            # Loading the model from raw bytes
            model = pickle.load(f)

            # Logging to inform that the model has been loaded
            message = f"model at {model_path} loaded successfully"
            self.fctp_file_operations_logging.info(message)

            return model

        except Exception as e:
            message = f"Error while loading model at {model_path}: {str(e)}"
            self.fctp_file_operations_logging.error(message)
            raise e

    def fctp_load_ml_model(self, cluster_no):
        """
        :Method Name: fctp_load_ml_model
        :Description: This method loads the ML model based on the cluster number.

        :param cluster_no: The number of the cluster based on which the model has to be selected and loaded
        :returns: The sklearn implemented ML model based on the cluster
        :On Failure: Exception
        """
        try:
            # Setting up the variable for ML Models path
            model_dir = "FCTPModels/FCTPMLModels/"

            # Going through all the models in the ML models Directory
            for filename in os.listdir(model_dir):
                if filename.endswith(f"_cluster_{cluster_no}.pickle"):
                    # Logging to infrom that the correct model has been selected
                    message = f"file: {filename} selected for prediction"
                    self.fctp_file_operations_logging.info(message)

                    # Returning the selected model
                    return self.fctp_load_model(os.path.join(model_dir, filename))

            # Logging to inform that no model was found
            message = "No Model Found"
            self.fctp_file_operations_logging.info(message)

        except Exception as e:
            message = f"Error while trying to retrieve data: {str(e)}"
            self.fctp_file_operations_logging.error(message)
            raise e
