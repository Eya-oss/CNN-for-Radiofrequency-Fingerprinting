# CNN-for-Radiofrequency-Fingerprinting
The project introduces a CNN machine learning model for Radiofrequency fingerprinting. The architecture of the model is highly inspired from ORACLE's papers , describing reliable radiofrequency fingerprinting techniques on high-end devices.
## **Presentation**
 This work makes use of the imperfections embedded within the phisical layer to implement a radiofrequency fingerprinting (RFF) technique in order to identify devices within a network.
Inspiring from previous works regarding RFF , this CNN-based algorithm was developed to classify devices  using the In-phase and Quadrature-phase samples recovered from the signals, and that because of their nature, cannot be completely identical from one device to another.  Indeed, this technique theoretically allows the identification of a device just through the analysis of its signal.
This model has achieved 99.9\% of accuracy on data transmitted over-the-air up to 5 devices. 
## **Structure**
This repository contains the following files:
- ***Data_load.py*** : where functions regarding data loading, converitng manipulation and pre-processing are developed.
- ***Model_CNN.py*** : where the model function is developed with respect to the number of outputs (number of devices to classify).
- ***Train_and_Evaluate.py*** : where functions of model training and predicting are developed. There are also functions of model evaluation through results displaying.
- ***main.py*** : This file contains an example of the use of the model and with all the data pre-processing operations.
