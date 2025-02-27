import cv2
import numpy as np
import matplotlib.pyplot as plt

class ObjectDetection:
    def __init__(self, model, classes, colors, image_path, transformations):
        self.model = model
        self.classes = classes
        self.colors = colors
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        self.width = self.original_image.shape[1]
        self.height = self.original_image.shape[0]
        self.transformations = transformations
        self.results = []
    
    def get_output_layers(self, net):
        layer_name = net.getLayerNames()
        output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layer
    
    def detect_objects(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), [0, 0, 0], 1, crop=False)
        self.model.setInput(blob)
        self.model.getLayerNames()
        outs = self.model.forward(self.get_output_layers(self.model))
        
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indexes) > 0 and isinstance(indexes, tuple):
            indexes = indexes[0]
        
        detections = []
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            confidence = confidences[i]
            detections.append((label, confidence, (x, y, w, h)))
        
        return detections

    def apply_transformations(self):
        transformations = self.transformations

        for transform_type, values in transformations.items():
            for value in values:
                # Reset the image for each transformation
                transformed_image = self.original_image.copy()

                if transform_type == 'size':
                    transformed_image = cv2.resize(
                        transformed_image, 
                        None, 
                        fx=value, 
                        fy=value, 
                        interpolation=cv2.INTER_AREA
                    )
                elif transform_type == 'rotation':
                    center = (transformed_image.shape[1]//2, transformed_image.shape[0]//2)
                    matrix = cv2.getRotationMatrix2D(center, value, 1)
                    transformed_image = cv2.warpAffine(
                        transformed_image, 
                        matrix, 
                        (transformed_image.shape[1], transformed_image.shape[0])
                    )
                elif transform_type == 'noise':
                    noise = value * np.random.randn(*transformed_image.shape)
                    transformed_image = transformed_image + noise
                    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)

                detections = self.detect_objects(transformed_image)
                if detections:
                    self.results.append((transform_type, value, detections))

    def plot_results(self):
        # Group results by transformation type
        grouped_results = {}
        for transform_type, value, detections in self.results:
            if transform_type not in grouped_results:
                grouped_results[transform_type] = {'values': [], 'confidences': []}
            avg_confidence = np.mean([conf for _, conf, _ in detections])
            grouped_results[transform_type]['values'].append(value)
            grouped_results[transform_type]['confidences'].append(avg_confidence)

        # Create a separate plot for each transformation type
        num_transformations = len(grouped_results)
        fig, axes = plt.subplots(num_transformations, 1, figsize=(10, num_transformations * 4))

        if num_transformations == 1:
            axes = [axes]  # Make it iterable if there's only one transformation

        for ax, (transform_type, data) in zip(axes, grouped_results.items()):
            ax.plot(data['values'], data['confidences'], label=f'{transform_type} transformation', marker='o')
            ax.set_xlabel(f'{transform_type} Values')
            ax.set_ylabel('Average Detection Confidence')
            ax.set_title(f'Detection Confidence Under {transform_type.capitalize()} Transformation')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def run(self):
        self.apply_transformations()
        self.plot_results()