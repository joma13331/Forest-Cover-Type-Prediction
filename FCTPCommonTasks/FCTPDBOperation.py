import os
import csv
import logging
import pandas as pd
import cassandra
from cassandra.query import dict_factory
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider


class FCTPDBOperation:
    """
    :Class Name: FCTPDBOperation
    :Description: This class will handle all the relevant operations related to Cassandra Database.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        """
        :Method Name: __init__
        :Description: This constructor initializes the variable that will be utilized
                      in all the class methods
        
        :param is_training: Boolean variable to inform whether training has to be done
        """

        # Checking if training is to be performed
        if is_training:
            # Creating training folder for logs if not already created
            if not os.path.isdir("FCTPLogFiles/training/"):
                os.mkdir("FCTPLogFiles/training/")

            # Variable which will tell in the logs whether log message is for training.
            self.operation = "TRAINING"

            # Setting up the Log File for the class FCTPDBOperation for training
            self.log_path = os.path.join("FCTPLogFiles/training", "FCTPDBOperation.txt")

            # Setting the GoodRaw Folder for training
            self.good_file_dir = "FCTPDIV/ValidatedData/GoodRaw/"

            # Table name to be used to store the data in Cassandra Database
            self.table_name = "good_training_data"
        else:
            # Creating prediction folder for logs if not already created
            if not os.path.isdir("FCTPLogFiles/prediction/"):
                os.mkdir("FCTPLogFiles/prediction/")

            # Variable which will tell in the logs whether log message is for prediction.
            self.operation = "PREDICTION"

            # Setting up the Log File for the class FCTPDBOperation for prediction
            self.log_path = os.path.join("FCTPLogFiles/prediction/", "FCTPDBOperation.txt")

             # Setting the GoodRaw Folder for training
            self.good_file_dir = "FCTPDIV/PredictionData/GoodRaw/"

            # Table name to be used to store the data in Cassandra Database
            self.table_name = "good_prediction_data"

        # Setting up the logging feature
        self.fctp_db_operation_logging = logging.getLogger("fctp_db_operation_log")
        self.fctp_db_operation_logging.setLevel(logging.INFO)
        fctp_db_operation_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_db_operation_handler.setFormatter(formatter)
        self.fctp_db_operation_logging.addHandler(fctp_db_operation_handler)

    def fctp_db_connection(self):
        """
        :Method Name: fctp_db_connection
        :Description: This method connects to the keyspace used for storing the validated
                      good dataset for this work.

        :return: session which is a cassandra database connection
        :On Failure: cassandra.cluster.NoHostAvailable, Exception
        """
        try:
            # storing the zip file to be used during Cassandra Database connection
            cloud_config = {
                'secure_connect_bundle': 'secure-connect-ineuron.zip'
            }
            # Providing the authentication for connecting to Cassandra Database
            auth_provider = PlainTextAuthProvider(os.getenv('CASSANDRA_CLIENT_ID'),
                                                  os.getenv('CASSANDRA_CLIENT_SECRET'))
            
            # Obtaining the cluster variable to connect to the cloud
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)

            # Establishing the connection to the Cassandra Database
            session = cluster.connect()
            # Setting up the session so the results are obtained as dictionary
            session.row_factory = dict_factory

            # Logging to inform that database connection has been established
            message = f"{self.operation}: Connection successful with cassandra database"
            self.fctp_db_operation_logging.info(message)

            # Using the keyspace for this project
            session.execute("USE forest_cover_type_prediction_internship;")

            # Logging to inform that the relevant keyspace has been accessed
            message = f"{self.operation}: accessed the forest_cover_type_prediction_internship keyspace"
            self.fctp_db_operation_logging.info(message)

            return session
        
        except cassandra.cluster.NoHostAvailable:
            message = f"{self.operation}: Connection Unsuccessful with cassandra database due to Incorrect " \
                      f"credentials or no connection from datastax"
            self.fctp_db_operation_logging.error(message)
            raise cassandra.cluster.NoHostAvailable
        except Exception as e:
            message = f"{self.operation}: Connection Unsuccessful with cassandra database: {str(e)}"
            self.fctp_db_operation_logging.error(message)
            raise e

    def fctp_create_table(self, column_names):
        """
        :Method Name: fctp_create_table
        :Description: This method creates a 'good_training_data' or 'good_prediction_data' table to store good data
                      with the appropriate column names.

        :param column_names: Column Names as expected from FCTPSchema based on DSA
        
        :return:None
        :On Failure: Exception
        """

        try:
            # obtaining the session variable for Cassandra Database
            session = self.fctp_db_connection()

            # Creating the table creation query
            table_creation_query = f"CREATE TABLE IF NOT EXISTS {self.table_name}(id int primary key,"
            for col_name in column_names:
                table_creation_query += f"\"{col_name}\" {column_names[col_name]},"
            # table_creation_query[:-1] is used to not consider the ',' at the end.
            table_creation_query = table_creation_query[:-1] + ");"
            print(table_creation_query)

            # Executing command to create the good data table
            session.execute(table_creation_query)

            # Logging to inform that the good data table was created
            message = f"{self.operation}: The table for Good Data created"
            self.fctp_db_operation_logging.info(message)

            # Ensuring that the table is cleared so that the latest data can be stored
            session.execute(f"truncate table {self.table_name};")

            # Logging that the data has been cleared
            message = f"{self.operation}: Any row if existing deleted"
            self.fctp_db_operation_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: The table for Good Data was Not created: {str(e)}"
            self.fctp_db_operation_logging.info(message)
            raise e

        finally:
            try:
                session.shutdown()
                message = f"{self.operation}: Session terminated create table operation"
                self.fctp_db_operation_logging.info(message)

            except Exception as e:
                pass

    def fctp_insert_good_data(self):
        """
        :Method Name: fctp_insert_good_data
        :Description: This method uploads all the files in the good Data folder
                      to the good_data tables in cassandra database.
        
        :return: None
        :On Failure: Exception
        """
        try:
            # obtaining the session variable for Cassandra Database
            session = self.fctp_db_connection()

            # This variable is used to run a command only once when value of count==0.
            count = 0

            # Setting up the col_names variable to store string of all columns which will
            # be used in cql query
            col_names = "id,"
            
            # Looping through all the files in the GoodRaw Directory
            for filename in os.listdir(self.good_file_dir):
                # Storing the data in the file into a temporary dataframe
                temp_df = pd.read_csv(os.path.join(self.good_file_dir, filename))

                # count variable is used so the the column part of the query is created only once as it is same for all
                # the insertion queries
                if count == 0:
                    for i in list(temp_df.columns):
                        col_names += f"\"{str(i).rstrip()}\","
                    # col_names[:-1] is used to not consider the ',' at the end
                    col_names = col_names[:-1]
                    count += 1

                # the for loop creates the values to be uploaded.
                # it is complicated to ensure that any 'null' value in a string column is entered as null and not a
                # simple string.
                for i in range(len(temp_df)):
                    
                    # [i+1] is the value for id.
                    temp_lis = [i+1] + list(temp_df.iloc[i])
                    # Checking if null is in the data row
                    if 'null' in temp_lis:
                        tup = "("
                        for j in temp_lis:
                            if type(j) == str:
                                if j == 'null':
                                    tup += f"{j},"
                                else:
                                    tup += f"'{j}',"
                            else:
                                tup += f"{j},"
                        tup = tup[:-1] + ")"
                    else:
                        tup = tuple(temp_lis)
                    # The insert query to add data to the Cassandra Database
                    insert_query = f"INSERT INTO {self.table_name}({col_names}) VALUES {tup};"
                    print(insert_query)
                    # Executing the insertion Query
                    session.execute(insert_query)
                # Logging to inform about information in a file has been uploaded
                message = f"{self.operation}: Data in {filename} uploaded successfully to good_data table"
                self.fctp_db_operation_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error while uploading data to good_data table: {str(e)}"
            self.fctp_db_operation_logging.error(message)
            raise e

        finally:
            try:
                session.shutdown()
                message = f"{self.operation}: Session terminated insert data operation"
                self.fctp_db_operation_logging.info(message)

            except Exception as e:
                pass

    def fctp_data_from_db_to_csv(self):
        """
        :Method Name: fctp_data_from_db_to_csv
        :Description: This method downloads all the good data from the cassandra
                      database to a csv file for preprocessing and training.
        
        :return: None
        :On Failure: Exception
        """
        try:
            # obtaining the session variable for Cassandra Database
            session = self.fctp_db_connection()

            # Checking to see if the operation is training or prediction
            if self.operation == 'TRAINING':
                # file to save the data from the database
                data_file = 'validated_file.csv'
                # Query to obtain the column names in the good_training_data table
                col_name_query = "select column_name from system_schema.columns where keyspace_name=" \
                                 "'forest_cover_type_prediction_internship' and table_name='good_training_data'; "
            else:
                # file to save the data from the database
                data_file = 'prediction_file.csv'
                # Query to obtain the column names in the good_prediction_data table
                col_name_query = "select column_name from system_schema.columns where keyspace_name=" \
                                 "'forest_cover_type_prediction_internship' and table_name='good_prediction_data'; "

            # List to store the headers for the data
            headers = []
            result = session.execute(col_name_query)
            for i in result:

                headers.append(str(i['column_name']))
                print(i['column_name'])

            # Query to obtain all the good data in the database 
            get_all_data_query = f"select * from {self.table_name};"
            # Storing all the good data in results which is a list of dictionaries
            # Because the session.rowfactory is dict_factory
            results = session.execute(get_all_data_query)

            # List to store all the data
            data = []

            for result in results:
                # List which stores all the information of a row
                row = []
                for header in headers:
                    # Since result is a dictionary with the column names as keys
                    row.append(result[header])
                data.append(row)

            # Opening the relevant csv files to storethe data into
            with open(data_file, 'w', newline='') as csv_file:
                # Obtaining the csv writer object to write into relevant csv file
                csv_writer = csv.writer(csv_file)
                # Writing the collumn names
                csv_writer.writerow(headers)
                # Writing the data
                csv_writer.writerows(data)

            # Logging to inform that the data from the database has been stored to a csv file
            message = f"{self.operation}: All data from good data table saved to {data_file}"
            self.fctp_db_operation_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error while downloading good data into csv file: {str(e)}"
            self.fctp_db_operation_logging.error(message)
            raise e

        finally:
            try:
                session.shutdown()
                message = f"Session terminated after downloading good data into csv file from table{self.table_name}"
                self.fctp_db_operation_logging.info(message)

            except Exception as e:
                pass

    def fctp_complete_db_pipeline(self, column_names, data_format_validator):
        """
        :Method Name: fctp_complete_db_pipeline
        :Description: This methods is written so that it can be run on a background thread to make ensure our web app
                      first makes the prediction to ensure less latency.
                      Only after the prediction is displayed on the web app does the database operations begin.
        
        :param column_names: The column names of the table in the cassandra database.
        :param data_format_validator: An object of FCTPDataFormatPred class to perform deletion and transfer of files
        
        :return: None
        :On Failure: Exception
        """
        try:
            # Creating the table in Cassandra database
            self.fctp_create_table(column_names=column_names)
            # Inserting the Data into the database
            self.fctp_insert_good_data()
            # Deleting the GoodRaw folder
            data_format_validator.fctp_delete_existing_good_data_folder()
            # Moving the files in BadRaw folder to archive
            data_format_validator.fctp_move_bad_files_to_archive()
            # Saving the data from the Cassandra database to a csv file
            self.fctp_data_from_db_to_csv()

        except Exception as e:
            message = f"{self.operation}: Error in Database Pipeline: {str(e)}"
            self.fctp_db_operation_logging.error(message)
            raise e