import os
import shutil
import threading
from datetime import datetime
from wsgiref import simple_server
from flask import Flask, render_template, url_for, request
from flask_cors import cross_origin, CORS
from wsgiref import simple_server

from FCTPCommonTasks.FCTPDataInjestionComplete import FCTPDataInjestionComplete
from FCTPPrediction.FCTPPredictionPipeline import FCTPPredictionPipeline
from FCTPTraining.FCTPTrainingPipeline import FCTPTrainingPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def fctp_home_page():
    # For displaying the iNeuron Logo
    img_url = url_for('static', filename='ineuron-logo.webp')
    # Rendering the Index/Home page
    return render_template('index.html', image_url=img_url)


@app.route("/train", methods=["POST"])
@cross_origin()
def fctp_train_route():
    # For displaying the iNeuron Logo
    img_url = url_for('static', filename='ineuron-logo.webp')

    try:
        # Checking to see if relevant information is submitted in the form
        if request.form is not None:
            # Obtaining the uploaded file
            file_item = request.files["dataset"]
            # Checking to see if the file_item has file name
            if file_item.filename:

                # Creating an empty folder to store the uploaded files
                if os.path.isdir("FCTPUploadedFiles"):
                    shutil.rmtree("FCTPUploadedFiles")
                    os.mkdir("FCTPUploadedFiles")
                else:
                    os.mkdir("FCTPUploadedFiles")

                # Saving the file
                with open(os.path.join("FCTPUploadedFiles", file_item.filename), 'wb') as f:
                    f.write(file_item.read())

                # Creating an object to perform data injestion and validation
                train_injestion_obj = FCTPDataInjestionComplete(is_training=True, data_dir="FCTPUploadedFiles")
                # Performing data injestion and validation
                train_injestion_obj.fctp_data_injestion_complete()

                # Creating an empty folder to save the models
                if os.path.isdir("FCTPModels"):
                    shutil.rmtree("FCTPModels")
                    os.mkdir("FCTPModels")
                else:
                    os.mkdir("FCTPModels")

                # Creating an object to perform the training
                training_pipeline = FCTPTrainingPipeline()
                # Training the model in a background thread so that the Web Application can remain active
                t2 = threading.Thread(target=training_pipeline.fctp_model_train)
                # Starting the training
                t2.start()

                # Rendering a Web Page to inform that the Training is going on 
                return render_template('train.html',
                                       message="Dataset validated.Training Started. Visit the web Application after "
                                               "1hr to perform prediction.", image_url=img_url)

            else:

                message = "No records Found\n TRY AGAIN"
                return render_template("train.html", message=message, image_url=img_url)
    except ValueError as e:
        return render_template('train.html', message=f"ERROR: {str(e)}\n TRY AGAIN", image_url=img_url)
        # raise e

    except KeyError as e:
        return render_template('train.html', message=f"ERROR: {str(e)}\n TRY AGAIN", image_url=img_url)
        # raise e

    except Exception as e:
        return render_template('train.html', message=f"ERROR: {str(e)}\n TRY AGAIN", image_url=img_url)
        # raise e


@app.route('/prediction', methods=["POST"])
@cross_origin()
def fctp_prediction_route():
    # For displaying the iNeuron Logo
    img_url = url_for('static', filename='ineuron-logo.webp')
    try:
        # Checking to see if relevant information is submitted in the form
        if request.form is not None:
            # Obtaining the uploaded file
            file_item = request.files["dataset"]
            # Checking to see if the file_item has file name
            if file_item.filename:
                
                # Creating an empty folder to store the uploaded files
                if os.path.isdir("FCTPUploadedFiles"):
                    shutil.rmtree("FCTPUploadedFiles")
                    os.mkdir("FCTPUploadedFiles")
                else:
                    os.mkdir("FCTPUploadedFiles")

                # Saving the file
                with open(os.path.join("FCTPUploadedFiles", file_item.filename), 'wb') as f:
                    f.write(file_item.read())

                # Creating an object to perform data injestion and validation
                pred_injestion_obj = FCTPDataInjestionComplete(is_training=False, data_dir="FCTPUploadedFiles",
                                                              do_database_operations=False)
                # Performing data injestion and validation
                pred_injestion_obj.fctp_data_injestion_complete()

                
                pred_pipeline = FCTPPredictionPipeline()
                # Performing Prediction
                result = pred_pipeline.fctp_predict()

                # Rendering a Web Page to display the results
                return render_template("predict.html", records=result, image_url=img_url)

            else:
                # A Message to inform regarding default 
                message = "Using Default FCTPPrediction Dataset"

                # Creating an object to perform data injestion and validation
                pred_injestion = FCTPDataInjestionComplete(is_training=False, data_dir="FCTPPredictionDatasets",
                                                           do_database_operations=False)
                # Performing data injestion and validation
                pred_injestion.fctp_data_injestion_complete()

                # Creating an object to perform the prediction
                pred_pipeline = FCTPPredictionPipeline()
                # Performing Prediction
                result = pred_pipeline.fctp_predict()

                return render_template("predict.html", message=message, records=result, image_url=img_url)

    except ValueError as e:
        message = f"Value Error: {str(e)}\nTry Again"
        return render_template("predict.html", message=message, image_url=img_url)

    except KeyError as e:
        message = f"Key Error: {str(e)}\nTry Again"
        return render_template("predict.html", message=message, image_url=img_url)

    except Exception as e:
        message = f"Error: {str(e)}\nTry Again"
        # return render_template("predict.html", message=message, image_url=img_url)
        raise e


@app.route("/logs", methods=["POST"])
@cross_origin()
def fctp_get_logs():
    img_url = url_for('static', filename='ineuron-logo.webp')
    try:
        if request.form is not None:
            log_type = request.form['log_type']

            with open(os.path.join("FCTPLogFiles/", log_type), "r") as f:
                logs = f.readlines()
            return render_template("logs.html", heading=log_type.split("/")[1], logs=logs, image_url=img_url)
        else:
            message = "No logs found"
            return render_template("logs.html", message=message, image_url=img_url)

    except Exception as e:
        message = f"Error: {str(e)}"
        return render_template("logs.html", heading=message, image_url=img_url)

port = int(os.getenv("PORT", 5000))

if __name__=="__main__":
    host = '0.0.0.0'
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
    