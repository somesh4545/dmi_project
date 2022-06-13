from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import csv
import tensorflow as tf

from flask_cors import CORS, cross_origin

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import image_utils


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img =  image_utils.load_img(img_path, target_size=(224, 224))

    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  
    input_arr = input_arr.astype('float32') / 255

    predictions = model.predict(input_arr)

    pre_class=predictions.argmax()

    return pre_class


@app.route('/', methods=['GET', "POST"])
def index():
    # Main page
    if request.method == 'POST':
        f = request.files['myFile']
        print(f)

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        preds = model_predict(file_path, model)

        result = preds

        if result == 0:
            result = "boys"
        elif result == 1:
            result = "girl"
        elif result == 2:
            result = "group"
        else:
            result = "pets"
        # result = "girl"
        file = open("Dataset/ClassData/Captions/"+result+".csv", "r") 
        # print("Dataset/ClassData/Captions/"+result,".csv")
        reader = csv.reader(file)
        i = 0
        captions = []
        for line in reader:
            if i!=0:
                obj = {
                    "caption": line[1],
                    "rating": line[2]
                }
                captions.append(obj)
            i+=1
            if i>=50:
                break

        response = {
            "class": result,
            "captions": captions,
            "total": i,
            "file": f
        }
        # response.headers.add("Access-Control-Allow-Origin", "*")
        return render_template('index.html', response=response)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)