# List to CNN

While trying to replicate large 2D convolutional networks, I realized it was getting a little tiresome, re-creating, copying-pasting a stack of layers again and again. 
So this is an attempt to create a template that allows you to just enter the parameters and the hyperparameters as a list or a dictionary object and it returns a list which can be:
1. Automatically adds ReLU layers to Conv and Dense Layers 
2. Edited to add or remove layers
3. Passed as an nn.Sequential() argument to create the model

We can use the template to configure how many times you want a single layer to be repeated or how many times a set of layers (stack) needs to be repeated

Currently written for torch. TF version coming soon !