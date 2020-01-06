# OCT_Classification_GAN
A Data-Efficient Approach for Automated Classification of OCT Images using Generative Adversarial Network

# Dependencies
Keras = 2.2.4 <br>
tensorflow-gpu = 1.9.0

Code should be compatible with Python versions 2.7-3.5. (tested in python2.7)

# Database
The database used to obtain the results is taken from the paper, R. Rasti, et al. "Macular OCT classification using a multi-scale convolutional neural network ensemble." IEEE transactions on medical imaging 37.4 (2017): 1024-1034.

For training purposes, the data should be divided into the train and validation set. A portion of the trained data (500 samples per class) is used as labeled data and should be placed in the following path: data/labeled <br>
The remaining trained data should be placed in: data/unlabeled

# Training 
The model can be trained with the command:
python train.py

# Testing
To reproduce the results of the paper, we have povided the learned model along with the validation set.
Use the following command to test the model:
python testModel.py

# Citation
If you use this code for research, please cite

@ARTICLE{8949716,
author={V. {Das} and S. {Dandapat} and P. K. {Bora}},
journal={IEEE Sensors Letters},
title={A Data-Efficient Approach for Automated Classification of OCT Images using Generative Adversarial Network},
year={2020},
volume={},
number={},
pages={1-1},
keywords={Sensor signals processing;classification;optical coherence tomography;generative adversarial network;semi-supervised learning},
doi={10.1109/LSENS.2019.2963712},
ISSN={2475-1472},
month={},}
