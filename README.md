Final Project Deep Learning NYU SPring 2024. 
Course Code: 
Instructor: 

This repository contains the code for semantic segmetnation of satellite images according to their building function. 
Building function has been derived from the landuse attribute in publicly available PLUTO Dataset of NYC, RGB satelite images have been taken from NAIP 2022. 

- models/UNetDilation.py represents the python file containing the final model architecture. Its based on a Unet with a dilation of 4 in the convolutional layer after the latent space. 
- training.ipynb contains the entire code for loading the training data, running the training and outputting the results along with validation. 
- best_model.pth and model_allEpochs.pth represent model checkpoints.
