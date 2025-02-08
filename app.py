import os
import sys
from flask import Flask, render_template, request, jsonify
from utils.predict import predict
from utils.classes import classes
from keras.models import load_model
import datetime
import statistics as st


model = load_model(os.path.join(os.getcwd(), "Models/bird_classification_model/birds_model2.h5"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/uploadFile", methods=["GET", "POST"])
def uploadFile():
    return render_template("uploadFile.html")

@app.route('/upload', methods=['POST'])
def upload_audio():
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d-%H:%M:%S")
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to a folder
    user_audio_folder = os.path.join(os.getcwd(), "UserData/Soundfile")
    
    filepath = os.path.join(user_audio_folder, file.filename)
    file.save(filepath)

    try:
        predicted_class = predict(model, now, filepath)
        predicted_class = st.mode(predicted_class)
        return jsonify({"prediction": classes[predicted_class], "text": "File uploaded successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8001)

