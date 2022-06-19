import logging
import os
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from FCTPCommonTasks.FCTPFileOperations import FCTPFileOperations


class FCTPClustering:
    """
    :Class Name: FCTPClusteringTrain
    :Description: This class is used to cluster the data so that models will be fine
                  tuned for each cluster and higher accuracy is obtained.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0 
    """

    def __init__(self):
        """
        :Method Name: __init__
        :Description: This constructor method initializes various variables
                      and objects used throughout the class.

        """

        # Variable which will tell in the logs whether log message is for training.
        self.operation = 'TRAINING'
        # Creating an object for saving the clustering model
        self.file_operator = FCTPFileOperations()

        # Setting up the folder for training logs
        if not os.path.isdir("FCTPLogFiles/training/"):
            os.mkdir("FCTPLogFiles/training/")

        # Setting up the path for the log file for this class
        self.log_path = "FCTPLogFiles/training/FCTPClustering.txt"

        # Directory where the model will be saved
        self.cluster_model_path = "FCTPModels/"

        # Setting up the logging feature
        self.fctp_clustering_logging = logging.getLogger("fctp_clustering_log")
        self.fctp_clustering_logging.setLevel(logging.INFO)
        fctp_clustering_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        fctp_clustering_handler.setFormatter(formatter)
        self.fctp_clustering_logging.addHandler(fctp_clustering_handler)


    def fctp_obtain_optimum_cluster_number(self, dataframe):
        """
        :Method Name: fctp_obtain_optimum_cluster_number
        :Description: This method calculates the optimum no. of cluster based on silhoette score

        :param dataframe: The dataframe representing the data from the client after
                          all the preprocessing has been done

        :return: The optimum cluster value
        :On Failure: Exception
        """
        try:
            
            # A dictionary with keys as cluster numbers and silhoette score as values
            sil_score = {}

            # Looping through all cluster values being checked
            for i in range(20,30):
                # Creating a Heirarchical Clustering object
                agl_clustering = AgglomerativeClustering(n_clusters=i)
                # Cluster labels for the data
                cluster_label = agl_clustering.fit_predict(dataframe)
                # Storing the silhoette scores into the dictionary
                sil_score[i] = silhouette_score(dataframe, labels=cluster_label)
            print(sil_score)
            
            # list of values of silhoette_scores
            values = list(sil_score.values())
            # list of cluster number being tested
            keys = list(sil_score.keys())

            # Index of the max Silhoette score
            index = values.index(max(values))
            # Optimum cluster number
            opt_cluster_val = keys[index]  

            # Logging about the optimum cluster number
            message = f"{self.operation}: The optimum cluster number is {opt_cluster_val}"
            self.fctp_clustering_logging.info(message)

            return opt_cluster_val

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while detecting optimum no. of cluster: {str(e)}"
            self.fctp_clustering_logging.error(message)
            raise e\
    
    def fctp_create_cluster(self, dataframe, number_of_clusters):
        """
        :Method Name: fctp_create_cluster
        :Description: This method performs the clustering in the dataset after preprocessing.

        :param dataframe: The pandas dataframe which has to be clustered.
        :param number_of_clusters: The number of clusters the data has to be clustered into.
        
        :return: Dataframe with clusters number added as a new column.
        :return: The sklearn Model used for clustering.
        :On Failure: Exception
        """
        try:
            
            # Creating a Heirarchical Cluster Model with optimum cluster number
            clustering_model = AgglomerativeClustering(n_clusters=number_of_clusters)

            # Adds a new column to the dataframe which identifies the cluster to which that data point belongs to.
            dataframe['cluster'] = clustering_model.fit_predict(dataframe)

            # Saving the clustering model
            self.file_operator.fctp_save_model(clustering_model, self.cluster_model_path, "cluster.pickle")

            # Logging that the clustering has been done
            message = f"{self.operation}: Clustering has been done, with cluster column added to dataset"
            self.fctp_clustering_logging.info(message)

            return dataframe

        except Exception as e:
            message = f"There was an ERROR while creating cluster: {str(e)}"
            self.fctp_clustering_logging.error(message)
            raise e