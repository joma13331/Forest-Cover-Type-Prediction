import json
import os
import re
import json
import shutil
import logging
import pandas as pd


class FCTPDataFormatValidator:
    """
    :Class Name: FCTPDataFormatValidator
    :Description: This class is used for handling all the data validation as needed
                  based on the agreement with the client.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training, path):
        """
        :Method Name: __init__
        :Description: This constructor method initializes variables for logging
                      It also sets up the path for storing Validated Data.
        :param is training: Whether this class is instantiated for training
        :param path: directory path where the files for training are present.
        """
        if is_training:

            # Setting up the folder for training logs
            if not os.path.isdir("FCTPLogFiles/training/"):
                os.mkdir("FCTPLogFiles/training/")

            # Setting up the path for the log file for this class
            self.log_path = os.path.join("FCTPLogFiles/training/", "FCTPDataFormatValidator.txt")

            # Variable which will tell in the logs whether log message is for training.
            self.operation = "TRAINING"

            # Make the ValidatedData Directory if absent
            if not os.path.isdir("FCTPDIV/ValidatedData/"):
                os.mkdir("FCTPDIV/ValidatedData/")

            # Storing the directory path where the Good and Bad Datasets are to be stored
            self.good_raw_path = "FCTPDIV/ValidatedData/GoodRaw/"
            self.bad_raw_path = "FCTPDIV/ValidatedData/BadRaw/"
            
            # Setting up the file name for the training schema
            self.schema_path = "FCTPSchemas/training_schema.json"
            # Setting up the name of csv file with all the validated data
            self.csv_filename = "validated_file.csv"
        
        else:

            # Setting up the folder for prediction logs
            if not os.path.isdir("FCTPLogFiles/prediction/"):
                os.mkdir("FCTPLogFiles/prediction/")
            
            # Setting up the path for the log file for this class
            self.log_path = os.path.join("FCTPLogFiles/prediction/", "FCTPDataFormatValidator.txt")

            # Variable which will tell in the logs whether log message is for training.
            self.operation = "PREDICTION"

            # Make the PredictionData Directory if absent
            if not os.path.isdir("FCTPDIV/PredictionData/"):
                os.mkdir("FCTPDIV/PredictionData/")

            # Storing the directory path where the Good and Bad Datasets are to be stored
            self.good_raw_path = "FCTPDIV/PredictionData/GoodRaw/"
            self.bad_raw_path = "FCTPDIV/PredictionData/BadRaw/"

            # Setting up the file name for the prediction schema
            self.schema_path = "FCTPSchemas/prediction_schema.json"
            # Setting up the name of csv file with all the validated data
            self.csv_filename = "prediction_file.csv"

        # Storing the directory path where the Datasets are Available
        self.dir_path = path

        #Setting up the logging feature
        self.fctp_data_format_validator_logging = logging.getLogger("fctp_data_format_validator_log")
        self.fctp_data_format_validator_logging.setLevel(logging.INFO)
        fctp_data_format_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_data_format_handler.setFormatter(formatter)
        self.fctp_data_format_validator_logging.addHandler(fctp_data_format_handler)

    def fctp_values_from_schema(self):
        """
        :Method Name: fctp_values_from_schema
        :Description: This method utilizes the json file in FCTPSchema from DSA to obtain
                      the expected dataset filename and dataset column details.

        :On Failure: can Raise ValueError, KeyError or Exception

        :return: 1. length of the Year that should be in filename
                 2. length of the Time that should be in filename
                 3. column names and corresponding datatype
                 4. Number of Columns expected in the dataset
        """

        try:
            # Accessing the schema into a dictionary
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)

            # Retrieving the Important information from the schema
            length_date_file = dic["LengthOfDate"]
            length_time_file = dic["LengthOfTime"]
            column_names = dic["ColumnNames"]
            column_number = dic["NumberOfColumns"]
            
            # Logging
            message = f"{self.operation}: Length of year in file = {length_date_file}\n" \
                f"Length of time ofin file = {length_time_file}\nNumber of columns = {column_number}"
            self.fctp_data_format_validator_logging.info(message)

            return length_date_file, length_time_file, column_names, column_number

        # Exception Handling
        except ValueError:
            message = f"{self.operation}: ValueError:Value not found inside schema_training.json"
            self.fctp_data_format_validator_logging.error(message)
            raise ValueError

        except KeyError:
            message = f"{self.operation}:KeyError:Incorrect key passed for schema_training.json"
            self.fctp_data_format_validator_logging.error(message)
            raise KeyError

        except Exception as e:
            self.fctp_data_format_validator_logging.error(f"{self.operation}: {str(e)}")
            raise e

    @staticmethod
    def fctp_regex_file_name():       
        """
        Method Name: fctp_regex_file_name
        Description: To generate the regex to compare whether the filename is
                     according to the DSA or not
        :return: Required Regex pattern
        :On Failure: None
        """
        # Creating a regex object to filter out files with incorrect names
        regex = re.compile(r'Forest_Cover_[0123]\d[01]\d[12]\d{3}_[012]\d[0-5]\d[0-5]\d.csv')
        return regex

    def fctp_create_good_bad_raw_data_directory(self):
        """
        :Method Name: fctp_create_good_bad_raw_data_directory
        :Description: This method creates directories to store the Good Data and Bad Data
                      after validating the training data.
        :return: None
        On Failure: OSError, Exception
        """
        try:
            
            # Creating the GoodRaw directory if it does not exist
            if not os.path.isdir(self.good_raw_path):
                os.makedirs(self.good_raw_path)
            # Creating the BadRaw directory if it does not exist
            if not os.path.isdir(self.bad_raw_path):
                os.makedirs(self.bad_raw_path)

            # Logging to inform about creation of the Folders
            message = f"{self.operation}: Good and Bad file directory created"
            self.fctp_data_format_validator_logging.info(message)

        except OSError as e:
            message = f"{self.operation}: Error while creating directory: {str(e)}"
            self.fctp_data_format_validator_logging.error(message)
            raise OSError

        except Exception as e:
            self.fctp_data_format_validator_logging.error(f"{self.operation}: {str(e)}")
            raise e

    def fctp_delete_existing_good_data_folder(self):
        """
        :Method Name: fctp_delete_existing_good_data_folder
        :Description: This method deletes the directory made to store the Good Data
                      after loading the data in the table. Once the good files are
                      loaded in the DB, deleting the directory ensures space optimization.
        :return: None
        :On Failure: OSError, Exception
        """
        try:
            # Deleting the GoodRaw directory if it exists
            if os.path.isdir(self.good_raw_path):
                shutil.rmtree(self.good_raw_path)

                #Logging to inform about deletion of GoodRaw folder
                message = f"{self.operation}: GoodRaw directory deleted successfully!!!"
                self.fctp_data_format_validator_logging.info(message)

        except OSError as e:
            message = f"{self.operation}: Error while creating directory: {str(e)}"
            self.fctp_data_format_validator_logging.error(message)
            raise e
        except Exception as e:
            self.fctp_data_format_validator_logging.error(f"{self.operation}: {str(e)}")
            raise e
    
    def fctp_delete_existing_bad_data_folder(self):
        """
        :Method Name: fctp_delete_existing_bad_data_folder
        :Description: This method deletes the directory made to store the Bad Data
                      after moving the data in an archive folder. We archive the bad
                      files to send them back to the client for invalid data issue.
        :return: None
        :On Failure: OSError
        """
        try:
            # Deleting the BadRaw directory if it exists
            if os.path.isdir(self.bad_raw_path):
                shutil.rmtree(self.bad_raw_path)

                #Logging to inform about deletion of BadRaw folder
                message = f"{self.operation}BadRaw directory deleted successfully!!!"
                self.fctp_data_format_validator_logging.info(message)

        except OSError as e:
            message = f"{self.operation}: Error while creating directory: {str(e)}"
            self.fctp_data_format_validator_logging.error(message)
            raise e
        except Exception as e:
            self.fctp_data_format_validator_logging.error(f"{self.operation}: {str(e)}")
            raise e
    
    def fctp_archive_bad_files(self):
        """
        :Method Name: fctp_archive_bad_files
        :Description: This method deletes the directory made to store the Bad Data
                      after moving the data in an archive folder. We archive the bad
                      files to send them back to the client for invalid data issue.
        :return: None
        : On Failure: Exception
        """

        try:
            # Checking to see if BadRaw folder exists 
            if os.path.isdir(self.bad_raw_path):

                # Creating the ArchivedData folder if needed
                archive_dir = "FCTPDIV/ArchivedData/"
                if not os.path.isdir(archive_dir):
                    os.mkdir(archive_dir)
                
                # obtaining list of all files present in the BadRaw Directory
                bad_files = os.listdir(self.bad_raw_path)
                print(bad_files)
                # Looping through every file in bad_files
                for file in bad_files:
                    # checking if the bad dile is already present in archive directory
                    archive_file_name = "_".join(["BadData"] + file.split("_")[-2:])
                    if archive_file_name not in os.listdir(archive_dir):
                        archive_file_path = os.path.join(archive_dir, archive_file_name)
                        shutil.move(self.bad_raw_path+file, archive_dir)
                        os.rename(os.path.join(archive_dir,file), archive_file_path)
                # Logging about tranfer of bad files to archive folder
                message = f"{self.operation}: ad files moved to archive: {archive_dir}"
                self.fctp_data_format_validator_logging.info(message)
                self.fctp_delete_existing_bad_data_folder()

        except Exception as e:
            message = f"{self.operation}: Error while Archiving Bad Files: {str(e)}"
            self.fctp_data_format_validator_logging.error(message)
            raise e

    def fctp_validating_filename(self, regex):
        """
        :Method Name: fctp_validating_file_name
        :Description: This function validates the name of the training csv files as per given name in the EESchema!
                      Regex pattern is used to do the validation.If name format do not match the file is moved
                      to Bad Raw Data folder else in Good raw data.
        :param regex: The regex compiler used to check validity of filenames
        :return: None
        :On Failure: Exception
        """ 
        try:
            # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
            self.fctp_delete_existing_bad_data_folder()
            self.fctp_delete_existing_good_data_folder()

            # create new directories
            self.fctp_create_good_bad_raw_data_directory()
            
            # Obtaining all the files in the directory path provided
            raw_files = [file for file in os.listdir(self.dir_path)]

            # Looping for all the files in the given directory
            for filename in raw_files:
                # Checking if the file name matches the regex agreed by the client
                if re.match(regex, filename):
                    # Copying the file to the GoodRaw Directory
                    shutil.copy(os.path.join(self.dir_path, filename), self.good_raw_path)
                    # Logging about tranfer of good file to the GoodRaw Path
                    message = f"{self.operation}: {filename} is valid!! moved to GoodRaw folder"
                    self.fctp_data_format_validator_logging.info(message)
                else:
                    # Copying the file to the BadRaw Directory
                    shutil.copy(os.path.join(self.dir_path, filename), self.bad_raw_path)
                    # Logging about tranfer of bad file to the BadRaw Path
                    message = f"{self.operation}: {filename} is not valid!! moved to BadRaw folder"
                    self.fctp_data_format_validator_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error occurred while validating filename: {str(e)}"
            self.fctp_data_format_validator_logging.error(message)
            raise e
                    
    def fctp_validate_column_length(self, number_of_columns):
        """
        :Method Name: fctp_validate_column_length
        :Description: This function validates the number of columns in the csv files.
                      It is should be same as given in the EESchema file.
                      If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                      If the column number matches, file is kept in Good Raw Data for processing.

        :param number_of_columns: The number of columns that is expected based on DSA
        :return: None
        :On Failure: OSERROR, EXCEPTION
        """
        try:
            # Looping through all the files in the GoodRaw Directory
            for filename in os.listdir(self.good_raw_path):
                # Opening the file as a pandas dataframe
                pd_df = pd.read_csv(os.path.join(self.good_raw_path, filename))

                # Accessing the number of columns in the relevant files by checking shape of the dataframe.
                if not pd_df.shape[1] == number_of_columns:
                    # As the number of columns in the file is not same as in schema move to BadRaw Directory.
                    shutil.move(os.path.join(self.good_raw_path, filename), self.bad_raw_path)
                    # Logging about the transfer of bad file to the BadRaw Directory
                    message = f"{self.operation}: invalid Column length for the file {filename}.File moved to Bad Folder"
                    self.fctp_data_format_validator_logging.info(message)
                else:
                    # Logging that the file has same number of columns as mentioned in the schema
                    message = f"{self.operation}: {filename} validated. File remains in Good Folder"
                    self.fctp_data_format_validator_logging.info(message)

            # Logging about all the files being validated with respect to number of column
            message = f"{self.operation}: Column Length Validation Completed!!"
            self.fctp_data_format_validator_logging.info(message)

        except OSError:
            message = f"{self.operation}: Error occurred when moving the file: {str(OSError)}"
            self.fctp_data_format_validator_logging.error(message)
            raise OSError
        except Exception as e:
            message = f"{self.operation}: Error occurred : {str(e)}"
            self.fctp_data_format_validator_logging.error(message)
            raise e

    def fctp_validate_every_column(self, column_names):
        """
        :Method Name: fctp_validate_every_column
        :Description: This method compares all the column names and datatypes present 
                      in the files in GoodRaw directory to the schema as agreed upon with 
                      the client.

        :param column_names: A dictionary containing the column name as key and
                             their datatypes as values
        :return: None
        :On Failure: Exception
        """
        try:
            # Looping through all the files in the GoodRaw Directory
            for filename in os.listdir(self.good_raw_path):
                # Opening the file as a pandas dataframe
                pd_df = pd.read_csv(os.path.join(self.good_raw_path, filename))
                # obtaining the list of columns present in the file
                column_names_df = pd_df.columns
                # obtaining a pandas series of column datatypes
                column_datatypes = pd_df.dtypes

                # Looping through all the column names from the schema
                for col in column_names:
                    # Checking whether the column name is present in the file
                    if col in column_names_df:
                        if not column_names[col] == str(column_datatypes[col])[:-2]:
                            # As the datatype of the column in the file is not same as in schema, move to BadRaw Directory.
                            shutil.move(os.path.join(self.good_raw_path, filename), self.bad_raw_path)
                            # Logging about the transfer of bad file to the BadRaw Directory
                            message = f"{self.operation}: invalid Column datatype for the column '{col}'.File moved to Bad Folder"
                            self.fctp_data_format_validator_logging.info(message)
                            break
                    else:
                        # As the column from the schema is not present in the file, move to BadRaw Directory.
                        shutil.move(os.path.join(self.good_raw_path, filename), self.bad_raw_path)
                        # Logging about the transfer of bad file to the BadRaw Directory
                        message = f"{self.operation}: column '{col}' absent in the file. File moved to Bad Folder"
                        self.fctp_data_format_validator_logging.info(message)
                        break

        except Exception as e:
            message = f"{self.operation}: Error occurred : {str(e)}"
            self.fctp_data_format_validator_logging.error(message)
            raise e       

    def fctp_validate_whole_columns_as_empty(self):
        """
        :Method Name: fctp_validate_whole_columns_as_empty
        :Description: This method validates that there are no columns in the given file
                      that has no values.
        :return: None
        :On Failure: OSError, Exception
        """
        try:
            # Looping through all the files in the GoodRaw Directory
            for filename in os.listdir(self.good_raw_path):
                # Opening the file as a pandas dataframe
                pd_df = pd.read_csv(os.path.join(self.good_raw_path, filename))
                
                for column in pd_df:
                    # checking whether all the columns are null.
                    if (len(pd_df[column]) - pd_df[column].count()) == len(pd_df[column]):
                        # As the whole column is empty move to BadRaw Folder
                        shutil.move(os.path.join(self.good_raw_path, filename), self.bad_raw_path)
                        # Logging about transfering the file to BadRaw folder
                        message = f"{self.operation}: invalid column {column}. Moving to Bad Folder"
                        self.fctp_data_format_validator_logging.info(message)
                        break
        except OSError:
            message = f"{self.operation}: Error occurred when moving the file: {str(OSError)}"
            self.fctp_data_format_validator_logging.error(message)
            raise OSError
        except Exception as e:
            message = f"{self.operation}: Error occurred : {str(e)}"
            self.fctp_data_format_validator_logging.error(message)
            raise e 
    
    def fctp_convert_direct_csv_to_csv(self):
        """
        :Method Name: fctp_convert_direct_csv_to_csv
        :Description: This function converts all the csv files which have been validated as being in the correct
                      format into a single csv file which is then used in preprocessing for training ML EEModels.
                      This function is used to improve the speed or latency of the web application as the app does not
                      have to wait for database operations before starting the training.
        :return: None
        :On Failure: Exception
        """
        try:
            # creating an empty list where pandas datframes will be appended to late concatenate
            # into on dataframe
            list_pd = []
            # Looping through all the files in the GoodRaw Folder
            for filename in os.listdir(self.good_raw_path):
                # Appending the dataframe to the list
                list_pd.append(pd.read_csv(os.path.join(self.good_raw_path, filename)))
            
            # Concatenating all the dataframe into one dataframe
            df = pd.concat(list_pd)

            # Converting the single dataframe into a csv file
            df.to_csv(self.csv_filename, header=True, index=False)
            # Logging about obtaining a single csv file with the validated data
            message = f"{self.operation}: Validated csv files Converted directly to required csv file for future preprocessing"
            self.fctp_data_format_validator_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error occurred while direct conversion from csv to csv: {str(e)}"
            self.fctp_data_format_validator_logging.info(message)
            raise e
