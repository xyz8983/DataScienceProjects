# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:17:46 2017

@author: xyz8983

step 1: normalize the data
step 2: reducing dimention using PCA
step 3: train neural network
step 4: predict

"""

import pandas as pd
import numpy
import self_defined_helper_functions as helper
from scipy.optimize import fmin_l_bfgs_b
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

#read training dataset and testing datast into panda dataframe
training_data_path = '../input/train.csv'
testing_data_path = '../input/test.csv'
training = pd.read_csv(training_data_path)   
testing_data = pd.read_csv(testing_data_path)
testing_data = testing_data.values
 
training_x = training.drop(['label'],axis='columns').values
training_y = training['label'].values 

#normalize data in order to perform PCA to reduce the data dimention
normalizer = Normalizer().fit(training_x)
training_x = normalizer.transform(training_x)
testing_data = normalizer.transform(testing_data)

pca = PCA(n_components=0.8, whiten=True).fit(training_x)
training_x = pca.transform(training_x)
testing_data = pca.transform(testing_data)


#design a Neural Network algorithm with 
#original pixels are 28*28 =784, but using PCA, the input layer dimention is reduced to 48,
#which greatly increase the speed of execution
#16 units on second layer, 10 unites on output layer
input_layer_size = 48
hidden_layer_size = 20
num_labels = 10

#initiate Theta:
Theta1 = helper.random_initialize(input_layer_size,hidden_layer_size) 
Theta2 = helper.random_initialize(hidden_layer_size,num_labels)
vector_theta = numpy.append(Theta1.flatten(), Theta2.flatten())

#train neural network
lda = 0.6
other_args = (input_layer_size, hidden_layer_size,num_labels, training_x, training_y, lda)
#using scipy.optimize.fmin_l_bfgs_b to optimize the cost function, get the trained thetas
trained_result = fmin_l_bfgs_b(helper.nnCostFunction, x0=vector_theta, args=other_args)
print("training is end")

trained_theta_vector = trained_result[0]
print("shape of trained theta", trained_theta_vector.shape)
trained_theta1 = numpy.reshape(trained_theta_vector[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), order='c')
trained_theta2 = numpy.reshape(trained_theta_vector[hidden_layer_size*(input_layer_size+1):],(num_labels, (hidden_layer_size+1)),order='c')

#predicting the label for testing dataset
predicted_label = helper.predict(trained_theta1, trained_theta2, testing_data)
print("prediction finished")

#output the predicting label along with image Id to a csv file
testing_y = pd.DataFrame(predicted_label, columns = ['Label'])
testing_y.index += 1
testing_y.index.name = 'ImageId'
testing_y.to_csv('testing_y_submission.csv', sep=',')


