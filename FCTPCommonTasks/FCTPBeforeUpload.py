import os
import logging
import pandas as pd

class FCTPBeforeUpload:
    """
    :Class Name: FCTPBeforeUpload
    :Description: This class is used to transform the Good Raw Files before uploading to
                  to cassandra database

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """
    def __init__(self, is_training):

        if is_training:
            # Setting the GoodRaw Folder for training
            self.good_raw_path = "FCTPDIV/ValidatedData/GoodRaw/"

            # Creating training folder for logs if not already created
            if not os.path.isdir("FCTPLogFiles/training/"):
                os.mkdir("FCTPLogFiles/training/")

            # Setting up the Log File for the class FCTPBeforeUpload for training
            self.log_path = "FCTPLogFiles/training/FCTPBeforeUpload.txt"

            # Variable which will tell in the logs whether log message is for training.
            self.operation = "TRAINING"
        else:
            # Setting the GoodRaw Folder for prediction
            self.good_raw_path = "FCTPDIV/PredictionData/GoodRaw/"

             # Creating prediction folder for logs if not already created
            if not os.path.isdir("FCTPLogFiles/prediction/"):
                os.mkdir("FCTPLogFiles/prediction/")

            # Setting up the Log File for the class FCTPBeforeUpload for prediction
            self.log_path = "FCTPLogFiles/prediction/FCTPBeforeUpload.txt"

            # Variable which will tell in the logs whether log message is for training.
            self.operation = "PREDICTION"

        # Setting up the logging feature
        self.fctp_before_upload_logging = logging.getLogger("fctp_before_upload_log")
        self.fctp_before_upload_logging.setLevel(logging.INFO)
        fctp_before_upload_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_before_upload_handler.setFormatter(formatter)
        self.fctp_before_upload_logging.addHandler(fctp_before_upload_handler)

    def fctp_replace_missing_with_null(self):
        """
        :Method Name: fctp_replace_missing_with_null
        :Description: This method replaces all the missing values with 'null'.
        :return: None
        :On Failure: Exception
        """

        try:

            # Find all the files in the acceptable files folder and fill 'null' wherever there are missing values.
            # 'null' is being used so that cassandra database can accept missing values even in numerical columns.

            # Looping through all the files available in the GoodRaw directory
            for filename in os.listdir(self.good_raw_path):
                # storing the file into a datframe
                temp_df = pd.read_csv(os.path.join(self.good_raw_path, filename))
                # Replacing all na values as 'null'
                temp_df.fillna('null', inplace=True)
                # saving the dataframe back to csv
                temp_df.to_csv(os.path.join(self.good_raw_path, filename), header=True, index=None)

                # Logging about the succesfull transformation
                message = f"{self.operation}: {filename} transformed successfully"
                self.fctp_before_upload_logging.info(message)

        except Exception as e:
            message = f"Data Transformation Failed: {str(e)}"
            self.fctp_before_upload_logging.error(message)
            raise e