# Motivation

## Why are we solving this problem?

With the current toll collection there has been a decreased amount of traffic flow and an increased amount of congestion due to the manual toll collection methods. To solve this problem not only means to move away from manual toll collection, but it means to increase traffic flow as well as decrese the amount of congestion that has been caused by the previous methods.

### Problem Statement

Traffic congestion and inefficiencies in toll collection on highways necessitate a modern, automated solution that can recognize license plates accurately under various conditions.

### Value Proposition

Deploying an ALPR system will streamline toll collection, reduce traffic bottlenecks, ensure accurate billing, and ultimately lead to better road management and traveler experience.

## Does an ML solution “fit” this problem?

An ML solution does fit this problem. This is beacuse we can make a ML algorithm that is dynamic and diverse which will allow for the automated toll collection, even in bad weather conditions, a speeding car, or even different license plate designs.

## Are we able to tolerate mistakes and some degree of risk?

While striving for high accuracy, the system must allow for a minimal error margin, recognizing that absolute perfection is unattainable. Mistakes must be managed through robust error-handling and customer support mechanisms.

## Is this solution feasible? 

The solution is feasible as ML has made major advancments in image recognition. However, we need to always be adding more data to the model and refining it so we can create as low of a miss rate as possible.

# Requirements

## Scope:

### What are out goals?

To develop and implement an ALPR system capable of recognizing license plates from various states at high speeds, in all lighting and weather conditions, for efficient and automated toll collection.

### What are the success criteria?

We have a few criteria to consider this successful. The major point would be a high accuracy in license plate recognition, this is because with a low accuracy we could accidentally toll the wrong cars. The other success criteria would be the need to be able to operate under diverse environment conditions.

## Requirements:

### What are our (system) Assumptions?

- A sufficiently diverse and comprehensive dataset for training the model is avaliable. 

### What are our (system) Requirements?

- Robust, high speed image capture technology. 
- Efficient and accurate ML models for image recognition.
- A way to toll the car that is recognized.

## Risk & Uncertanties:

### What are the possible harms?

- Misidentification leading to incorrect billing
- Privacy concerns regarding data handling and storage
- System downtime affecting toll collection

### What are the causes of mistakes?

- Insufficiently trained models
- Poor image quality due to hardware limitations or environmental conditions
- Software bugs or data processing errors

# Implementation

## DEVELOPMENT:

### Methodology

Utilizing ML techniques such as deep learning or ensemble methods to help the image recognition algorithms understand any license plate.

### High-level System Design

Ingest a video file (in actual system this would be a live stream of cars going by). Chop the video into frames. Run model to detect which frams contain a car. Chop the frame into a small subset that only contianes the license plate. Pass into the OCR. Get the text output to send to toll collection program.

### Development Workflow

- Dataset preparation and augmentation
- Model training and evaluation using the provided yolov3 and yolov3-tiny architectures


### Data Engineering Pipeline

The Data Engineering Pipeline is a crucial component of the Automated License Plate Recognition (ALPR) system. It is designed to handle the initial data extraction, perform necessary transformations, and prepare the data for modeling. The pipeline consists of three primary methods: `extract()`, `transform()`, and `load()`, which streamline the process from raw data collection to ready-to-use data for the object detection model.

#### Extract

The `extract()` method automates the retrieval of image data from the source directory. It efficiently scans the directory, identifying and compiling a list of image files. This method is tailored to recognize common image file formats, including JPEG and PNG, ensuring compatibility and ease of integration with the diverse dataset of vehicle images.

#### Transform

The `transform()` method is responsible for applying a series of image transformations to prepare the data for the object detection model. This includes:

- Grey scaled the image for less noise.
- Applied a Gaussian Noise as well to reduce noise in the image.
- A sharpen method has been applied for easier reading of the images.

These transformations not only enhance the model's ability to generalize across different real-world conditions but also help in mitigating overfitting by providing a more robust dataset.

#### Load

After the images have been processed, the `load()` method ensures that the augmented dataset is saved and organized within a specified target directory. This method not only creates the directory if it does not exist but also systematically names and stores the processed images to facilitate easy access and identification.

By automating these processes, the Data Engineering Pipeline effectively reduces manual overhead, minimizes the risk of human error, and ensures a consistent and standardized preparation of data for subsequent stages of the ALPR system development.

### Dataset Partitioning Strategy

For the evaluation of our pre-trained models within the Automated License Plate Recognition (ALPR) system, a critical step involves partitioning the preprocessed dataset to ensure a comprehensive and unbiased assessment. The dataset, consisting solely of .png images of license plates, has been meticulously prepared to facilitate this process.

#### Object_Detection_Dataset Class

We introduce the `Object_Detection_Dataset` class, a pivotal component of our data handling framework, designed to manage the dataset's partitioning. This class incorporates functionalities for splitting the dataset into distinct sets, enabling effective model evaluation without the necessity for training within this project's scope.

##### Design and Functionality

- **Initialization and Data Loading**: Upon instantiation, the class loads the dataset from a specified directory, ensuring all data is accessible for partitioning.

- **Dataset Splitting**: The core of the class lies in its ability to divide the dataset into validation and testing subsets. Given the absence of model training in this project, these partitions allow for a thorough evaluation of the pre-trained models under different conditions and scenarios.

- **K-Fold Cross-Validation**: To bolster the evaluation's robustness, the class is equipped with a k-fold cross-validation method. This approach provides a rigorous framework for assessing model performance across multiple subsets, ensuring a well-rounded evaluation that mirrors a wide array of real-world applications.

##### Rationale

The decision to forego a traditional training set in favor of focusing on validation and testing stems from the project's unique constraints — namely, the provision of pre-trained models. This strategy ensures that the evaluation process remains focused on the models' ability to generalize to unseen data, a critical factor for real-world deployment. Additionally, the inclusion of k-fold cross-validation offers a methodical and reproducible means of quantifying model performance, further solidifying the validity of our evaluation process.

#### Implementation

The `Object_Detection_Dataset` class's implementation details, including its methods for dataset loading, partitioning, and k-fold cross-validation, are meticulously documented in the `dataset.py` module. This documentation provides clear guidance on leveraging the class for dataset preparation and model evaluation, ensuring transparency and ease of replication for future iterations of the ALPR system.

By adopting this partitioning strategy, we aim to establish a benchmark for model evaluation that is both rigorous and aligned with the project's overarching goals. This approach underscores our commitment to deploying a robust and reliable ALPR system capable of meeting the demands of modern toll collection and traffic management.

### Metrics Pipeline

Our Automated License Plate Recognition (ALPR) system's performance is meticulously evaluated to ensure its reliability and efficiency. The cornerstone of our evaluation strategy is the `Metrics` class within the `metrics.py` module. This class embodies a structured methodology for calculating and reporting critical performance indicators, which now include Average Intersection over Union (IoU), Precision, Recall, and the F1 Score, alongside an insightful model of overall effectiveness. These metrics are instrumental in assessing object detection tasks integral to ALPR technologies.

#### Metrics Class Implementation

The `Metrics` class facilitates a comprehensive analysis by employing the following metrics:

- **Average IoU:** Represents the average overlap between the predicted bounding boxes and the ground truth, with values closer to 1 indicating higher accuracy in localization.
- **Precision:** Indicates the proportion of positive identifications that were actually correct, crucial for minimizing false positives in detection scenarios.
- **Recall:** Measures the ability of the model to detect all relevant instances, vital for ensuring no vehicle goes undetected.
- **F1 Score:** Provides a harmonized metric that balances precision and recall, reflecting the overall accuracy of the detection process.

Furthermore, we introduced a novel approach to gauge the model's overall effectiveness, reflecting a holistic view of its performance across various aspects of detection and localization.

#### Generating Reports

The metrics pipeline culminates in the generation of detailed reports through the `run` function within the `Metrics` class. This function processes the model's predictions against the ground truths to compute the outlined metrics. A comprehensive report summarizing these findings is then automatically generated and stored within a designated `results` directory, facilitating easy access and review.

Incorporating this enhanced metrics pipeline into our ALPR system underscores our commitment to deploying a solution characterized by high accuracy, reliability, and user trust. Systematic evaluation empowers us to continuously refine our model, ensuring it meets the high standards required for effective license plate recognition.

## POLICY:

### Human-Machine Interfaces

Interfaces for monitoring system performance, managing billing discrepancies, and customer service will be developed to handle exceptions and support users.

### Regulations

Compliance with data protection and privacy laws, as well as transportation regulations, will be ensured through legal consultation and policy development.

## OPERATIONS:

### Continuous Deployment

Checking to ensure that the system is not charging the wrong license plate. 

### Post-deployment Monitoring

A close eye needs to be kept to ensure that the system is not charging the wrong license plate. 

### Maintenance

Any maintanence would be to adjust how the image is being sent into pytesseract

### Quality Assurance

Tests were completed to get a good model for reading license plates.