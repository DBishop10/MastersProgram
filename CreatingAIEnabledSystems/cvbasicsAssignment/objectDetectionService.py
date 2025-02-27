from flask import Flask
from flask import request
from flask import jsonify
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from GraphicDataProcessing import ObjectDetection

app = Flask(__name__)

# Use postman to generate the post with a graphic of your choice

classes = []
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
transformations = {
            'size': [0.1, 0.5, 1, 1.5, 2, 5, 10],
            'rotation': [0, 90, 180, 270],
            'noise': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        }
colors = np.random.uniform(0, 255, size=(len(classes), 3))  # Color for each class
model = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # Load the pre-trained model

@app.route('/detect', methods=['POST'])
def detection():
    args = request.args
    name = args.get('name')
    location = args.get('description')
    
    imagefile = request.files.get('imagefile', '')
    print("Image: ", imagefile.filename)
    imagefile.save('Pictures/' + imagefile.filename)
    # The file is now downloaded and available to use with your detection class
    ot = ObjectDetection(model, classes, colors, ('Pictures/' + imagefile.filename), transformations)
    findings = ot.detect_objects(ot.original_image)
    # covert to useful string
    findingsString = jsonify(findings)
    os.remove('Pictures/' + imagefile.filename)  
    return findingsString

if __name__ == "__main__":
    flaskPort = 8786
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

