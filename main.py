import os
from flask import Flask, render_template
from flask_cors import cross_origin, CORS
from wsgiref import simple_server


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def fctp_home_page():
    return render_template("index.html", message="Setting up the CI/CD Pipeline")

port = int(os.getenv("PORT", 5000))

if __name__=="__main__":
    host = '0.0.0.0'
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
    