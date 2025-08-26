import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from End_to_End_ML_Pipeline.utils.common import decode_image
from End_to_End_ML_Pipeline.pipeline.predict import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.file_name = "input_image.jpg"
        self.classifier = PredictionPipeline(self.file_name)
        
@app.route("/", methods = ['GET'])
@cross_origin()
def home_page():
    return render_template("index.html")

@app.route("/train", methods = ['GET', 'POST'])
@cross_origin()
def train_route():
    os.system("dvc repro")
    return "Training done successfully!!"

@app.route("/predict", methods = ['POST'])
@cross_origin()
def predict_route():
    image = request.json['image']
    decode_image(image, clApp.file_name)
    result = ClientApp().classifier.predict()
    return jsonify(result)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host = '0.0.0.0', port = 8080)