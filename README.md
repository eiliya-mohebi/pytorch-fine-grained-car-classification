# PyTorch Fine-Grained Car Classification
<div align="center">
  <img src="./demo-gif.gif" alt="App Demo" width="600"/>
</div>

This repository contains a PyTorch implementation for the task of fine-grained car classification using the Stanford Cars dataset. The goal is to identify the specific make, model, and year of a car from an image.

## Project Objective

The primary objective of this project was to build a car classification system capable of achieving **Accuracy, Precision (Avg), and Recall (Avg) metrics all above 75%** on the validation set.

## Approach

The solution is built using deep learning and leverages transfer learning from a powerful pre-trained convolutional neural network.

1.  **Dataset:** Utilized the [Stanford Cars dataset](https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars), accessed conveniently via the Kaggle API.
2.  **Model:** Employed **EfficientNet-B3**, a state-of-the-art CNN known for its balance of efficiency and accuracy, pre-trained on ImageNet.
3.  **Fine-tuning:** The entire pre-trained EfficientNet-B3 model was fine-tuned on the Stanford Cars dataset, allowing the model to adapt its powerful features to the nuances of the car images.
4.  **Custom Classifier:** Replaced the original classifier head with a custom one designed to the dataset's specific number of classes, incorporating `BatchNorm1d` and `Dropout` for stability and regularization.
5.  **Regularization:** Techniques like increased **Dropout (p=0.7)** in the classifier and **Weight Decay (L2 regularization)** in the optimizer were used to combat overfitting.
6.  **Data Augmentation:** Applied extensive data augmentation (including random crops, flips, color jitter, rotation) to the training data to improve model generalization.
7.  **Mixed Precision Training:** Utilized `torch.cuda.amp` for **Automatic Mixed Precision (AMP)** training, speeding up the training process and reducing GPU memory usage.
8.  **Early Stopping:** Implemented **Early Stopping** based on validation accuracy to halt training when performance on unseen data plateaued, preventing severe overfitting.

## Results

After training with the implemented strategies, the model achieved the following metrics on the validation dataset at its best epoch:

* **Validation Accuracy:** **~82.66%**
* **Validation Precision (Avg):** **~83.29%**
* **Validation Recall (Avg):** **~82.62%**

All key metrics successfully **exceeded the 75% target**, demonstrating good performance on this fine-grained classification task.

## Potential Improvements & Future Work

While the project met its requirements, further improvements could be explored to achieve even higher accuracy:

* **Hyperparameter Tuning:** Systematically tune learning rates, optimizer parameters, weight decay, dropout rates, and augmentation parameters.
* **Larger EfficientNet Models:** Experiment with EfficientNet-B4, B5, or larger variants. Be careful that larger models require more resources and may be more prone to overfitting, needing stronger regularization and careful tuning.
* **Explore Other Architectures:** Consider other high-performing CNNs or Vision Transformers.


## Acknowledgements

* [Stanford Cars Dataset](https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars)
* [EfficientNet](https://arxiv.org/abs/1905.11946)
