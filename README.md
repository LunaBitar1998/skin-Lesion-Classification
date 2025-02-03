# skin-Lesion-Classification
This repository contains a modular and flexible framework for skin lesion classification, designed to facilitate experimentation with different deep learning models, hyperparameters, and training configurations. The input dataset consists of dermoscopic images categorized into two classes: nevus and others.

## Dataset Summary
The dataset used for this project includes dermoscopic images from three widely recognized datasets: HAM10000, BCN_20000, and MSK Dataset. The dataset comprises:

Training Dataset: 15,195 images
Validation Dataset: 3,796 images
## Purpose of the Repository
The main purpose of this repository is to streamline the process of training deep learning models for binary classification of dermoscopic images. The modular structure allows users to:

1. Experiment with Different Models: Users can easily switch between models such as ConvNeXt, EfficientNet, DenseNet, etc., by simply changing the model name in the configuration file.
2. Test Various Hyperparameters: Loss functions, optimizers, dropout rates, and other hyperparameters can be adjusted without modifying the core training code.
3. Ensemble Multiple Models: The framework includes functionality for ensemble predictions, allowing users to boost performance by combining the outputs of different models.
4. Facilitate Future Research: By making the training and testing process easier to configure, this framework encourages further experimentation to identify the best-performing model and hyperparameter combinations.
## Achieved Results
Baseline Model: The ConvNeXt-Tiny model achieved an accuracy of 90% on the validation set.
Future Work:
Additional combinations of models and hyperparameters will be tested to further improve performance.
The ensemble method implemented in this framework is expected to boost overall accuracy.
Metrics such as precision, recall, and F1-score will also be evaluated for a more comprehensive performance analysis.

## Project Structure


 ├── src/
│   ├── config.py         
│   ├── data_loader.py 
│   ├── ensemble.py       
│   ├── models.py         
│   ├── optimizers.py    
│   ├── test.py           
│   ├── training.py       
│   ├── transforms.py  
│   ├── utils.py         
├── README.md          
├── main.py               
├── requirements.txt      

