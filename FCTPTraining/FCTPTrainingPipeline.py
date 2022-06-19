import os
import logging

from sklearn.model_selection import train_test_split

from FCTPCommonTasks.FCTPFileOperations import FCTPFileOperations
from FCTPCommonTasks.FCTPDataLoader import FCTPDataLoader
from FCTPCommonTasks.FCTPEda import FCTPEda
from FCTPCommonTasks.FCTPFeatureEngineering import FCTPFeatureEngineering
from FCTPCommonTasks.FCTPFeatureSelection import FCTPFeatureSelection
from FCTPTraining.FCTPClustering import FCTPClustering
from FCTPTraining.FCTPModelFinderTrain import FCTPModelFinderTrain


class FCTPTrainingPipeline:
    """
    :Class Name: FCTPTrainingPipeline
    :Description: This class contains the methods which integrates all the relevant classes and their methods
                  to perform data preprocessing, training and saving of the best model for later predictions.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self):
        """
        :Method Name: __init__
        :Description: This constructor method sets up the variables that will be used throughout
                      this class
        """
        # # Variable which will tell in the logs whether log message is for training.
        self.operation = 'TRAINING'
        
        # Setting up the folder for training logs
        if not os.path.isdir("FCTPLogFiles/training/"):
            os.mkdir("FCTPLogFiles/training")
        # Setting up the path for the log file for this class
        self.log_path = os.path.join("FCTPLogFiles/training/", "FCTPTrainingPipeline.txt")

        # Creating the directory where all the ML models will be stored
        if not os.path.isdir("FCTPModels/FCTPMLModels/"):
            os.mkdir("FCTPModels/FCTPMLModels/")
        # Directory where all the ML models will be stored
        self.ml_model_dir = "FCTPModels/FCTPMLModels/"

        # Creating the directory, if not already created where general models will be stored
        if not os.path.isdir("FCTPModels/"):
            os.mkdir("FCTPModels/")
        # Directory where all the General models will be stored    
        self.cluster_dir = "FCTPModels/"

        # Creating the directory, if not already created where all the relevant will be stored
        if not os.path.isdir("FCTPRelInfo/"):
            os.mkdir("FCTPRelInfo/")
        # Directory where all the relevant information will be stored
        self.rel_info_dir = "FCTPRelInfo/"
        
        # Setting up the logging feature
        self.fctp_training_pipeline_logging = logging.getLogger("fctp_training_pipeline_log")
        self.fctp_training_pipeline_logging.setLevel(logging.INFO)
        fctp_training_pipeline_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_training_pipeline_handler.setFormatter(formatter)
        self.fctp_training_pipeline_logging.addHandler(fctp_training_pipeline_handler)

    def fctp_model_train(self):
        """
        :Method Name: fctp_model_train
        :Description: This method integrates all the relevant classes and their methods to perform
                      Data Preprocessing, Clustering and saving the best model for each of the cluster.
        :return: None
        :On Failure: Exception
        """

        try:

            # Logging about Start Of Training Pipeline
            message = f"{self.operation}: Start of Training Pipeline"
            self.fctp_training_pipeline_logging.info(message)

            # Logging about getting Validated data
            message = f"{self.operation}: Getting Validated Data"
            self.fctp_training_pipeline_logging.info(message)

            # GETTING THE DATA
            data_loader = FCTPDataLoader(is_training=True)
            validated_data = data_loader.fctp_get_data()

            # Logging about having gotten Validated data
            message = f"{self.operation}: Validated Data Obtained"
            self.fctp_training_pipeline_logging.info(message)

            # DATA PRE-PROCESSING
            # Logging about start of Data Preprocessing
            message = f"{self.operation}: Data Preprocessing started"
            self.fctp_training_pipeline_logging.info(message)

            # Initializing The objects needed for Data Preprocessing
            eda = FCTPEda(is_training=True)
            feature_engineer = FCTPFeatureEngineering(is_training=True)
            feature_selector = FCTPFeatureSelection(is_training=True)
            file_operator = FCTPFileOperations()

            # Removing The 'id' and 'Id' column
            temp_df = feature_selector.fctp_remove_columns(validated_data, ['id', 'Id'])
            message = f"{self.operation}: Removed the 'id' and 'Id' column"
            self.fctp_training_pipeline_logging.info(message)

            # Splitting the data into features and label
            features, label = eda.fctp_feature_label_split(temp_df, ['Cover_Type'])
            message = f"{self.operation}: Separated the features and labels"
            self.fctp_training_pipeline_logging.info(message)

            # Checking whether null values are present and in what columns they are present
            is_null_present, columns_with_null = eda.fctp_features_with_missing_values(features)

            # Creating alist to store columns to be dropped
            col_to_drop = []

            # Checking if null values are present
            if is_null_present:
                features, dropped_features = feature_engineer.fctp_handling_missing_data_mcar(features,
                                                                                             columns_with_null)
                col_to_drop.extend(dropped_features)

            # Logging about imputed values
            message = f"{self.operation}: Checked for null values and if any were present imputed them"
            self.fctp_training_pipeline_logging.info(message)

            # obtaining the numerical and categorical features
            numerical_feat, categorical_feat = eda.fctp_numerical_and_categorical_columns(features)
            message = f"{self.operation}: Obtained the Numerical and Categorical Features"
            self.fctp_training_pipeline_logging.info(message)

            # Obtaining the continuous and discrete numerical features
            cont_feat, discrete_feat = eda.fctp_continuous_discrete_variables(features, numerical_feat)
            message = f"{self.operation}: Obtained the Continuous and Discrete Features"
            self.fctp_training_pipeline_logging.info(message)

            # Adding features with zero Standard deviation to the list
            col_to_drop.extend(feature_selector.fctp_features_with_zero_std(features))
            # Adding features which are not important to the list 
            col_to_drop.extend(feature_selector.fctp_feature_not_important(features=features, label=label, cat_cols=discrete_feat, threshold=0.005))
            
            # Removing repeated values in the list  
            col_to_drop = list(set(col_to_drop))
            # Converting the list of columns to be dropped to string to store in a txt file
            col_to_drop_str = ",".join(col_to_drop)

            # Writing the column to be dropped into a text file
            with open(os.path.join(self.rel_info_dir, "columns_to_drop.txt"), 'w') as f:
                f.write(col_to_drop_str)

            # Removing columns that have to be dropped
            features = feature_selector.fctp_remove_columns(features, col_to_drop)
            message = f"{self.operation}: Dropped all the irrelevant columns after feature selection"
            self.fctp_training_pipeline_logging.info(message)

            # Converting features to more normal if possible
            features = eda.fctp_obtain_normal_features(features, cont_feat)
            message = f"{self.operation}: Converted all possible continuous columns to normal"
            self.fctp_training_pipeline_logging.info(message)

            # Performing Standard Scaling on the features
            features = feature_engineer.fctp_standard_scaling_features(features)
            message = f"{self.operation}: All the features have been scaled"
            self.fctp_training_pipeline_logging.info(message)
            
            # Feature Reduction using 
            features = feature_engineer.fctp_pca_decomposition(features, variance_to_be_retained=0.95)
            message = f"{self.operation}: Performed PCA and retained 99% of variance"
            self.fctp_training_pipeline_logging.info(message)

            message = f"{self.operation}: Data Preprocessing completed"
            self.fctp_training_pipeline_logging.info(message)

            # CLUSTERING

            message = f"{self.operation}: Data Clustering Started"
            self.fctp_training_pipeline_logging.info(message)

            cluster = FCTPClustering()
            num_clusters = cluster.fctp_obtain_optimum_cluster_number(features)
            features = cluster.fctp_create_cluster(features, num_clusters)

            features['Cover_Type'] = label

            list_of_cluster = features['cluster'].unique()

            message = f"{self.operation}: Data Clustering Completed"
            self.fctp_training_pipeline_logging.info(message)

            # Training of Each Cluster

            for i in list_of_cluster:
                message = f"{self.operation}: Start of Training for cluster {i}"
                self.fctp_training_pipeline_logging.info(message)

                cluster_data = features[features['cluster'] == i]
                cluster_feature = cluster_data.drop(columns=['Cover_Type', 'cluster'])
                cluster_label = cluster_data['Cover_Type']

                train_x, test_x, train_y, test_y = train_test_split(cluster_feature, cluster_label, random_state=42)
                train_x = train_x
                test_x = test_x
                model_finder = FCTPModelFinderTrain()
                model_name, model = model_finder.fctp_best_model(train_x=train_x, train_y=train_y,
                                                                test_x=test_x, test_y=test_y)

                file_operator.fctp_save_model(model=model, model_dir=self.ml_model_dir,
                                             model_name=f"{model_name}_cluster_{i}.pickle")

                message = f"{self.operation}:Model for cluster {i} trained"
                self.fctp_training_pipeline_logging.info(message)

            message = f"{self.operation}: Successful End of Training "
            self.fctp_training_pipeline_logging.info(message)

            message = f"{self.operation}: Training Pipeline Successfully Completed"
            self.fctp_training_pipeline_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: There was an ERROR in obtaining best model: {str(e)}"
            self.fctp_training_pipeline_logging.info(message)
            raise e