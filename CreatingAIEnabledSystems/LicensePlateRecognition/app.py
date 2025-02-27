from model import Object_Detection_Model
from data_pipeline import Pipeline
import pytesseract
import os

def process_frame(frame, predictions):
    # Modify frame based on predictions
    # This is where you can draw bounding boxes, for instance
    for prediction in predictions:
        prediction = prediction.replace('(', '').replace(')', '').split(',')
        bounding_box = [int(prediction[1]), int(prediction[2]), int(prediction[3]), int(prediction[4])]
        cropped_image = dp.crop_license_plate(format_type="yolov3output", yolov3Output=bounding_box, yolov3image=frame)
        cleaned_image = dp.transform(cropped_image, True)
        predicted_license_plates = pytesseract.image_to_string(cleaned_image, config ='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 11 --oem 1')
        if predicted_license_plates != " ":
            print("Predicted License Plate: " + predicted_license_plates)
    
if __name__ == '__main__':
    # Instantiate and use the model
    model = Object_Detection_Model()
    dp = Pipeline()
    udp_url = os.getenv('UDP_URL')
    if not udp_url:
        udp_url = 'LicensePlateReaderSample_4k.mov'
        raise ValueError("No UDP URL provided")

    model.process_video_stream(udp_url, process_frame)