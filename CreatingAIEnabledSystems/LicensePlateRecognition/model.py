import cv2
import numpy as np
from metrics import Metrics
from data_pipeline import Pipeline
import matplotlib.pyplot as plt

class Object_Detection_Model:
    def __init__(self):
        # Initialize and load your YOLOv3 model
        self.model = self.load_model()

    def load_model(self):
        model = cv2.dnn.readNetFromDarknet('Models/lpr-yolov3.cfg', 'Models/lpr-yolov3.weights')
        return model
    
    def test(self, predictions, ground_truth, model="yolov3"):
        """
        Processes the image with the object detection model and returns predictions.
        Parameters:
        predictions: What is the predicted bounding box
        ground_truth: What is the actual bounding box
        model: what model type it is, specificall for report name, default yolov3
        """
        metrics = Metrics()
        metrics.run(predictions, ground_truth, report_name="object_detection_report_" + model + ".txt")
        
    def predict(self, image, size=(416, 416), conf_threshold=0.7, nms_threshold=0.4):
        """
        Processes the image with the object detection model and returns predictions.
        Parameters:
        image: The image to process.
        size: Size of said image, default (416, 416)
        conf_threshold: confidence threshold for the model, default 0.7
        nms_threshold: nms threshold for the model, default 0.4
        Return: 
        prediction_location_and_class: Predictions made by the model.
        """
        # Load the image from the path
        # Convert the preprocessed image to a blob
        blob = cv2.dnn.blobFromImage(image, 1/255.0, size, swapRB=True, crop=False)
        self.model.setInput(blob)

        # Get the output layer names
        layer_names = self.model.getLayerNames()
        output_layers_indices = self.model.getUnconnectedOutLayers().flatten()
        output_layers = [layer_names[i - 1] for i in output_layers_indices]

        # Forward pass to get output of the output layers
        outputs = self.model.forward(output_layers)
    
        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x, center_y, width, height = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, int(width), int(height)])

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        final_predictions = []
        prediction_location_and_class = []
        if len(indices) > 0:
        # Convert tuple of arrays to a list of indices if necessary
            if isinstance(indices[0], np.ndarray):
                indices = [idx[0] for idx in indices]
            # Otherwise, use indices directly if it's already in the desired format (list of integers)

            for i in indices:
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                final_predictions.append((class_ids[i], confidences[i], (x, y, w, h)))
        for prediction in final_predictions:
            prediction_location_and_class.append("" + str(prediction[0]) + ", " + str(prediction[2]))
        if prediction_location_and_class != []:
            return prediction_location_and_class

        
    def process_video_stream(self, video_path, callback):
        """
        processes the video stream
        Parameters:
        video_path: The path of the video to process
        callback: callback function to send the processed frame back to app.py
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Make predictions on the current frame
            predictions = self.predict(frame)
            if predictions is not None:
                callback(frame, predictions)

        cap.release()