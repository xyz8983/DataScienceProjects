## Steps:
1. Read MNIST data into pandas data frame
2. Normalize data in order to use PCA
3. Reduce the data dimension by PCA, otherwise there are two many input units for the Neural Network algorithm. My laptop easily died if I did not reduce the dimension.
4. Train 3-layer Neural Network
     * input layer: 48 units, because the PCA returns 48 dimension
     * hidden layer: 20 unites, I just select a tolerable number for my laptop to do the computation
     * output layer: 10 unites, because there are 10 digit for recognition
     * I write a cost function with gradient descent, optimize it using scipy.optimize.fmin_l_bfgs_b, then get the trained parameter Theta.
5. Predict the label for testing data set using the trained Neural Network, write output to csv file.
