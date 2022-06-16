import os
import logging
import numpy as np
import pandas as pd

class FCTPDataLoader:
    """
    :Class Name: FCTPDataLoader
    :Description: This class contains the method for loading the data into
                  a pandas dataframe for future usage

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):

        # Checking whether the object has been initialized for training or prediction
        if is_training:

            # Variable which will tell in the logs whether log message is for training.
            self.operation = 'TRAINING'
            # file from which the dataframe has to created
            self.data_file = 'validated_file.csv'
            # Creating training folder for logs if not already created
            if not os.path.isdir("FCTPLogFiles/training/"):
                os.mkdir("FCTPLogFiles/training/")
            # Setting up the Log File for the class FCTPDataLoader for training    
            self.log_path = "FCTPLogFiles/training/FCTPDataLoader.txt"

        else:

            # Variable which will tell in the logs whether log message is for training.
            self.operation = 'PREDICTION'
            # file from which the dataframe has to created
            self.data_file = 'prediction_file.csv'

            # Creating prediction folder for logs if not already created
            if not os.path.isdir("FCTPLogFiles/prediction/"):
                os.mkdir("FCTPLogFiles/prediction/")
            # Setting up the Log File for the class FCTPDataLoader for training    
            self.log_path = "FCTPLogFiles/prediction/FCTPDataLoader.txt"

        # Setting up the logging feature
        self.fctp_dataloader_logging = logging.getLogger("fctp_dataloader_log")
        self.fctp_dataloader_logging.setLevel(logging.INFO)
        fctp_dataloader_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_dataloader_handler.setFormatter(formatter)
        self.fctp_dataloader_logging.addHandler(fctp_dataloader_handler)

    def fctp_get_data(self):
        """
        :Method Name: fctp_get_data
        :Description: This method reads the data from source.

        :returns: A pandas DataFrame.
        :On Failure: Raise Exception
        """
        try:
            # the object variable which contains the relevant dataframe
            self.data = pd.read_csv(self.data_file)

            # To round all the values to two decimal digits as it is usually in the data files.
            self.data = self.data.round(2)
            
            #Logging to inform that the dataframe has been obtained
            message = f"{self.operation}: The data is loaded successfully as a pandas dataframe"
            self.fctp_dataloader_logging.info(message)
            
            return self.data

        except Exception as e:
            message = f"{self.operation}: Error while trying to load the data for prediction to pandas dataframe: {str(e)}"
            self.fctp_dataloader_logging.error(message)
            raise e