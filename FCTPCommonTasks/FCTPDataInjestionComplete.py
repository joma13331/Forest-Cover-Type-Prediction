import os
import logging
import threading

from FCTPCommonTasks.FCTPDataFormatValidator import FCTPDataFormatValidator
from FCTPCommonTasks.FCTPBeforeUpload import FCTPBeforeUpload
from FCTPCommonTasks.FCTPDBOperation import FCTPDBOperation


class FCTPDataInjestionComplete:
    """
    :Class Name: FCTPDataInjestionComplete
    :Description: This class utilized 3 Different classes
                    1. FCTPDataFormatTrain
                    2. FCTPBeforeUploadTrain
                    3. FCTPDBOperationTrain
                    to complete validation on the dataset names, columns, etc based on
                    the DSA with the client. It then uploads the valid files to a cassandra
                    Database. Finally it obtains a csv from the database to be used further
                    for preprocessing and training

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training, data_dir, do_database_operations=False):
        """
        :Method Name: __init__
        :Description: This constructor method initializes the variables that will be used in methods of this class.

        :param is_training: Whether this class is instantiated for training.
        :param data_dir: Data directory where files are present.
        :param do_database_operations: Whether database operations have to be carried out.
        """

        self.data_format_validator = FCTPDataFormatValidator(is_training=is_training, path=data_dir)
        self.db_operator = FCTPDBOperation(is_training=is_training)
        self.data_transformer = FCTPBeforeUpload(is_training=is_training)

        # Check to seeif the object has been instantiated for training or prediction
        if is_training:

            # Variable which will tell in the logs whether log message is for training.
            self.operation = 'TRAINING'

            # Creating training folder for logs if not already created
            if not os.path.isdir("FCTPLogFiles/training/"):
                os.mkdir("FCTPLogFiles/training")
            # Setting up the Log File for the class FCTPDataInjestionComplete for training
            self.log_path = os.path.join("FCTPLogFiles/training/", "FCTPDataInjestionComplete.txt")
        
        else:

            # Variable which will tell in the logs whether log message is for prediction.
            self.operation = 'PREDICTION'

            # Creating prediction folder for logs if not already created
            if not os.path.isdir("FCTPLogFiles/prediction/"):
                os.mkdir("FCTPLogFiles/prediction")
            # Setting up the Log File for the class FCTPDataInjestionComplete for prediction
            self.log_path = os.path.join("FCTPLogFiles/prediction/", "FCTPDataInjestionComplete.txt")

        # Creating a class variable which will ensure that database operations will be carried or not.
        self.do_database_operation = do_database_operations
        
        # Setting up the logging feature
        self.fctp_data_injestion_logging = logging.getLogger("fctp_data_injestion_log")
        self.fctp_data_injestion_logging.setLevel(logging.INFO)
        fctp_data_injestion_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_data_injestion_handler.setFormatter(formatter)
        self.fctp_data_injestion_logging.addHandler(fctp_data_injestion_handler)    

    def fctp_data_injestion_complete(self):
        """
        :Method Name: fctp_data_injestion_complete
        :Description: This method is used to complete the entire data validation,
                      data injestion process to store the data in a database and
                      convert it for further usage in our project work

        :return: None
        :On Failure: Exception
        """
        try:

            # Logging to inform about the start of Data Injestion and Validation
            message = f"{self.operation}: Start of Injestion and Validation"
            self.fctp_data_injestion_logging.info(message)

            # Obtaining the relevant information from the schema
            length_date, length_time, dataset_col_names, dataset_col_num = self.data_format_validator.\
                fctp_values_from_schema()
            # Obtaining the regex object which will validate the file names
            regex = self.data_format_validator.fctp_regex_file_name()
            # Validating the filenames
            self.data_format_validator.fctp_validating_filename(regex)
            # Validation the number of columns
            self.data_format_validator.fctp_validate_column_length(dataset_col_num)
            # Validating all the column names and types
            self.data_format_validator.fctp_validate_every_column(column_names=dataset_col_names)
            # Validating whether a complete column is empty
            self.data_format_validator.fctp_validate_whole_columns_as_empty()
            # Moving all the files in bad folder to archive
            self.data_format_validator.fctp_archive_bad_files()

            # Logging about Data Validation being completed
            message = f"{self.operation}: Raw Data Validation complete"
            self.fctp_data_injestion_logging.info(message)

            # Logging about start of Data Trandformation
            message = f"{self.operation}: Start of Data Transformation"
            self.fctp_data_injestion_logging.info(message)

            # Replacing nan values by 'null' so that it can be stored in cassandra database
            self.data_transformer.fctp_replace_missing_with_null()

            # Logging about Completion of data transformation
            message = f"{self.operation}: Data Transformation Complete"
            self.fctp_data_injestion_logging.info(message)

            # Logging about start of DB operations
            message = f"{self.operation}: Start of upload of the Good Data to Cassandra Database"
            self.fctp_data_injestion_logging.info(message)

            # Checking to see if database operations need to be carried out
            if self.do_database_operation:

                # Threading used to bypass time consuming database tasks to improve web application latency.
                t1 = threading.Thread(target=self.db_operator.fctp_complete_db_pipeline,
                                      args=[dataset_col_names, self.data_format_validator])
                t1.start()
                # t1 not joined so that it runs only after training has occurred.

            # Converting all the file in GoodRaw folder to single csv file for further
            # preprocessing
            self.data_format_validator.fctp_convert_direct_csv_to_csv()

            # Logging about Completion of Data Injestion and Validation
            message = f"{self.operation}: End of Injestion and Validation"
            self.fctp_data_injestion_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error During Injestion and Validation Phase{str(e)}"
            self.fctp_data_injestion_logging.error(message)
            raise e