from flask import Flask,render_template,request
from prediction_services import predictions
import os
from werkzeug.utils import secure_filename
import yaml
from tensorflow.keras.models import load_model


webapp_root="webapp"
static_dir=os.path.join(webapp_root,"static")
template_dir=os.path.join(webapp_root,"templates")

app=Flask(__name__,static_folder=static_dir,template_folder=template_dir)

@app.route("/",methods=['GET','POST'])

def home():
    return render_template("index.html")



params_path="config/params.yaml"


def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config


config=read_params(params_path)
MODEL_PATH = config['model_dir']
model = load_model(MODEL_PATH)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        return predictions.model_predict(file_path,model)

    return "not working"

if __name__=="__main__":
    app.run(host="127.0.0.1",port=9090,debug=True)