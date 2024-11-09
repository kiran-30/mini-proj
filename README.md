**Tomato Leaf Health Classification:**

This project uses machine learning to classify tomato leaf images as healthy or as affected by various diseases. By detecting tomato leaf health issues early, farmers and gardeners can make informed decisions to improve crop yields and reduce loss.


**Project Overview**:

The goal of this project is to classify tomato leaves into various categories using Convolutional Neural Networks (CNNs). The model classifies leaves as healthy or diseased with different types of tomato diseases, such as Bacterial Spot, Early Blight, Late Blight, and Leaf Mold, among others.



**Dataset**:

The dataset used consists of categorized tomato leaf images, which are divided into training and validation folders:
Training Data: For model training, with augmented images to enhance model robustness.
Validation Data: For model evaluation, to ensure the model generalizes well.



**Technologies and Libraries Used**:

Programming Languages: Python
Libraries:
TensorFlow/Keras: For building and training the CNN model
NumPy: For numerical operations
Matplotlib: For plotting model performance metrics
pathlib and os: For handling file paths
ImageDataGenerator: For data augmentation and preprocessing



**Model Architecture**:

The CNN model consists of:
Convolutional layers for feature extraction with relu activation
MaxPooling layers for down-sampling
Dropout layer for regularization
Dense layers with softmax for classification



**Data Augmentation**:

Data augmentation techniques applied on the training images include:
Rotation, zoom, width/height shift, and horizontal flip
These augmentations help the model generalize better to unseen data.



**Training and Evaluation**:

The model was trained with categorical cross-entropy loss and adam optimizer, using accuracy as the primary metric. Model performance is evaluated on the validation set and visualized through accuracy and loss plots.



**Prediction Example**:

To classify a new tomato leaf image:
Load and preprocess the image.
Use the trained model to predict the class.
Output the class label as either "Healthy" or a specific disease type.



**Results**:

The model achieved an accuracy of 98% on the validation set.
