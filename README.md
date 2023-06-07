# SegSalad - A Hands On Review of Semantic Segmentation Techniques for Weed/Crop Datasets

The source code for this project is available in the [SegSalad.ipynb](SegSalad.ipynb) Jupyter Notebook. You can also access and run the notebook on Google Colab [here](https://colab.research.google.com/drive/1Xmzz54j1JgksESurdbqAeGQFGnHiCQU9?usp=sharing).

## Abstract

This project explores various semantic segmentation techniques used for weed/crop datasets in precision agriculture. The project includes training three different semantic segmentation neural network architectures using transfer learning, employing image patching as a data augmentation technique, and evaluating the models qualitatively and with the Intersection Over Union (IOU) metric. The challenges and key findings of the project are discussed. For a video summary of the project, click [here](https://youtu.be/FyGz-Pb-K2k).

## Problem and Motivation

Precision agriculture aims to increase crop yield while reducing resource usage. Computer vision plays a vital role in achieving this goal by enabling targeted pesticide application, automated harvesting, and plant health monitoring. This project focuses on semantic segmentation of weeds and crops, which is particularly relevant for targeted pesticide application.

## Dataset and Data Augmentation

The dataset used in this project is the CWFID Carrot/Weed image dataset (1). It contains 60 high-resolution images of carrot plants and weeds from an organic carrot farm. The dataset provides pixel-level annotations, with the red channel representing weeds, the green channel representing crops, and the blue channel representing the background. To overcome the high resolution and limited dataset size, image patching is applied, dividing the images into smaller patches of size 224x224. This data augmentation technique improves training efficiency. The dataset is split randomly into an 80% training set and a 20% testing set.

## Semantic Segmentation Models

Three semantic segmentation models are trained in this project: Fully Convolutional Network (FCN), Google DeepLabv3, and UNet. Each model utilizes a ResNet50 backbone. Transfer learning is employed by utilizing pretrained models. The FCN and DeepLabv3 models are pretrained on the COCO 2017 dataset, while the UNet model is pretrained on ImageNet. Mean Squared Error (MSE) loss is used during training, and the IOU metric is used for model evaluation.

## Model Evaluation

The models are evaluated using the Intersection Over Union (IOU) metric. The following table shows the IOU results for crops, weeds, and soil:

| Model         | Crop IOU | Weed IOU | Soil IOU |
| ------------- | -------- | -------- | -------- |
| FCN           | 42.79%   | 66.78%   | 97.66%   |
| DeepLabv3     | 20.32%   | 67.55%   | 97.86%   |
| UNet          | 49.45%   | 75.21%   | 98.38%   |

The UNet model outperforms the FCN and DeepLabv3 models for all classes. However, all models struggle with accurately detecting crops, while being relatively good at detecting weeds and soil. It is worth noting that overfitting may have occurred due to fine-tuning for too many epochs.

## Takeaways and Future Work

This project provides insights into common semantic segmentation models, training neural networks in PyTorch, and their application in precision agriculture. Future work could involve exploring additional network architectures, training models from scratch, and incorporating domain-specific knowledge into the models. Further improvements could be made by using larger datasets with more diverse weed and crop types.

## References

(1) P. Ribera et al., "Crop/Weed Field Images Dataset (CWFID): A Large Dataset for Semantic Segmentation," in IEEE Robotics and Automation Letters, vol. 4, no. 3, pp. 2290-2297, July 2019, doi: 10.1109/LRA.2019.2897353.
