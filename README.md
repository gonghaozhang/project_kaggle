# Bird Species Classification using VGG19
The source code for this project is available in the form of a Jupyter notebook, which can be run in a colab, anaconda or kaggle environment with the necessary libraries installed.

## Abstract
This project focuses on creating a deep learning model for bird species classification. The model architecture is based on VGG19, a model known for its effectiveness in image recognition tasks. The model takes bird images as input and predicts the species of the bird.

## Dataset
The dataset is organized into a training set and a testing set, both of which include images of various bird species. Each bird species has its own directory, and all images of that species are contained within that directory. The dataset path needs to be set before running the model.

## Image Preprocessing
All images are resized to 224x224 pixels, as VGG19 works best with this image size. ImageDataGenerator, a tool from the Keras library, is used to automatically generate batches of tensor image data with real-time data augmentation. The data is normalized by dividing every pixel in the image by 255 (the maximum value), so that each pixel is in the range [0, 1].

## Model Architecture
The base of the model is VGG19, pre-trained on the ImageNet dataset. We remove the top layer, which is responsible for classifying the 1000 ImageNet classes. Instead, a new layer is added that matches the number of bird species in the dataset. This layer uses the softmax activation function, as we are dealing with a multi-class classification problem.

## Training
The model is compiled with Adam optimizer and categorical cross entropy loss function. The model is then trained for 10 epochs, with the data being passed in batches.

## Evaluation and Visualization
The model is evaluated using the testing set, providing us with its accuracy. We also plot graphs showing how the model's accuracy and loss change over time for both the training and validation sets. This allows us to visualize how well our model is learning and whether it is overfitting or underfitting.

![accuracy and loss](accuracy.png)

## Prediction
The model is then used to predict the species of the birds in the testing set. It outputs probabilities for each bird species, and the species with the highest probability is chosen as the prediction. We generate a classification report to evaluate the model’s performance.

We also generate a confusion matrix to see how well the model has performed for each bird species. The diagonal elements represent the number of points for which the predicted label is equal to the true label, while off-diagonal elements are those that are mislabeled by the classifier.

## Usage
You need to set the correct dataset path for train_data_dir and test_data_dir before running the script. The script is written in Python and requires certain libraries. Please ensure that the following libraries are installed in your Python environment:

- os
- cv2
- pickle
- itertools
- numpy
- pandas
- seaborn
- matplotlib
- tensorflow
- sklearn
- Future Work
<br>

Although the model shows decent results, there are several ways it can be improved. The accuracy might increase with a larger, more diverse dataset. Hyperparameter tuning could further improve model performance. Different architectures like ResNet or EfficientNet could be tested for better results.