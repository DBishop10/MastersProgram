import xml.etree.ElementTree as ET
import os
import json
import numpy as np
import cv2
import ffmpeg
import random

class Pipeline:
    def extract(self, source_directory):
        """Extracts image files from the specified directory."""
        # Get all file names in the directory
        file_list = os.listdir(source_directory)
        # Filter out files that are not images (assuming JPEG and PNG formats)
        image_files = [os.path.join(source_directory, f) for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return image_files

    def transform(self, image_path, yoloImage=False):
        """Change image for use of training"""
        if not yoloImage:
            image = self.crop_license_plate(image_path)
            image_id = image_path.split('/')[1].split('.')[0]
        else:
            image = image_path
        if image is None:
            print(f"Failed to read image: {image_path}")
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply a slight blur to reduce noise
            blur = cv2.GaussianBlur(gray, (3,3), 0)

            # Apply thresholding to get a binary image
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Sharpen the image
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(binary, -1, kernel)
            if not yoloImage:
                return sharpened, image_id
            else:
                return sharpened
    
    def random_brightness_contrast(self, image):
        """Randomly adjusts the brightness and contrast of the image."""
        brightness = random.uniform(0.5, 1.5)
        contrast = random.uniform(0.5, 1.5)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return image

    def random_affine(self, image):
        """Applies a random affine transformation to the image."""
        rows, cols, ch = image.shape
        
        # Random scale and rotation
        scale = random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale)
        
        # Random translation
        tx = random.uniform(-cols * 0.1, cols * 0.1)
        ty = random.uniform(-rows * 0.1, rows * 0.1)
        M[0, 2] += tx
        M[1, 2] += ty

        image = cv2.warpAffine(image, M, (cols, rows))
        return image

    def random_hsv(self, image):
        """Randomly adjusts the hue, saturation, and value of the image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        hue_shift = random.randint(-10, 10)
        h = cv2.add(h, hue_shift)
        sat_shift = random.uniform(0.7, 1.3)
        s = cv2.multiply(s, sat_shift)
        val_shift = random.uniform(0.7, 1.3)
        v = cv2.multiply(v, val_shift)

        hsv_image = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return image
    
    def crop_license_plate(self, image_path="", format_type="coco", yolov3Output="", yolov3image=""):
        if image_path != "":
            image = cv2.imread(image_path)
            image_name = image_path.split('/')[1]
            img_width = image.shape[1]
            img_height = image.shape[0]
    
        if format_type == "coco":
            with open('combined_coco_annotations.json', 'r') as j:
                data = json.loads(j.read())
            image_id = None
            for image_data in data["images"]:
                if image_data["file_name"] == image_name:
                    image_id = image_data["id"]
                    break
            if image_id is not None:
                bboxes = []
                for annotation in data["annotations"]:
                    if annotation["image_id"] == image_id:
                        x_min, y_min, width, height = annotation["bbox"]
                        return image[y_min:y_min+height, x_min:x_min+width]

        elif format_type == "yolo":
            with open('combined_yolo_annotations.txt', 'r') as j:
                contents = j.readlines()
            for line in contents:
                if(line.split(',')[0] == image_name.split('.')[0]):
                    contents = line.split(', ')
            contents.pop(0)
            contents.pop(0)
            contents[3]= contents[3].split("\n")[0]
            x_center, y_center, width, height = map(float, contents)
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            return image[y_min:y_min+int(height), x_min:x_min+int(width)]
        elif format_type == "yolov3output":
            (x, y, w, h) = yolov3Output
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(yolov3image.shape[1], x + w)
            y_max = min(yolov3image.shape[0], y + h)
            return yolov3image[y_min:y_max, x_min:x_max]
    
    def load(self, images, target_directory):
        """Loads the processed images into the target directory."""
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)  # Create target directory if it doesn't exist

        for image in images:
            if image is None:
                print(f"Failed to write image: {image[1]}")
            else:
                file_name = f"processed_image_{image[1]}.png"
                file_path = os.path.join(target_directory, file_name)
                cv2.imwrite(file_path, image[0])
            
    def convert_pascal_to_yolo_format(xml_content):
        tree = ET.parse(xml_content)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        yolo_annotations = []

        for obj in root.iter('object'):
            class_label = obj.find('name').text

            class_id = 0 if class_label == 'number_plate' else -1  # Placeholder for actual class id mapping

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            yolo_format = f"{class_id}, {x_center:.6f}, {y_center:.6f}, {bbox_width:.6f}, {bbox_height:.6f}"
            yolo_annotations.append(yolo_format)

        return yolo_annotations

    def convert_pascal_to_coco_format(xml_file):
        """Parses a single PASCAL VOC XML file and returns COCO format annotation."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        number = root.find('filename').text.split(".")[0]
        imageid = number[1:]
        coco_format = {
            "info": {
                "description": "Dataset",
                "version": "1.0",
                "year": 2024
            },
            "licenses": [
                {
                    "id": imageid,
                    "name": "Licenses from Teacher",
                    "url": "https://jhu.instructure.com/courses/66216/files/10066954?wrap=1"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": imageid,
                    "name": "number_plate",
                    "supercategory": "vehicle"
                }
            ]
        }

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        coco_format["images"].append({
            "id": imageid,
            "width": width,
            "height": height,
            "file_name": filename,
            "license": imageid,
            "date_captured": "2024-03-08 10:00:00"
        })

        ann_id = 1

        for obj in root.iter('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            o_width = xmax - xmin
            o_height = ymax - ymin
            area = o_width * o_height

            coco_format["annotations"].append({
                "id": ann_id,
                "image_id": imageid,
                "category_id": 1,  
                "segmentation": [],
                "area": area,
                "bbox": [xmin, ymin, o_width, o_height],
                "iscrowd": 0
            })
            ann_id += 1

        return coco_format