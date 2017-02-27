# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:12:43 2017

@author: sylvia

this file contains all self-defined functions 
"""
import numpy


def sigmoid(z):
    return 1/(1+numpy.exp(-z))

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lda):
    """
    calculating cost function for Neural Network
    the cost function is similar to the one for logistic regression
    input: 
        nn_params: a 1-D numpy.ndarray, all thetas flattened in a vector
        input_layer_size: integer, the size of first layer of neural network, 
        pixels number of one image here, notated as n
        hidden_layer_size: integer, the size of second layer of neural network,
        notated as h
        num_labels: integer, the size of output layer, notated as l
        X: 2-D numpy.ndarray, the training dataset
        y: 1-D numpy.ndarray, the actual lables for each dataset
        lda: float, lambda, for regularization purpose
    output: float, the cost of the neural network model
        
    """
    #part1: cost function:
    
    #only 2 Thetas because only three layer for this neural network   
    Theta1 = numpy.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), order='c')
    Theta2 = numpy.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels, (hidden_layer_size+1)),order='c')
    (m, n) =  X.shape
    J = 0
    Theta1_grad =  numpy.zeros(Theta1.shape)
    Theta2_grad =  numpy.zeros(Theta2.shape)
    
    a1 = numpy.append(numpy.ones((m,1)), X, axis=1)  # m * (n+1)
    binary_y = numpy.zeros((m,num_labels))   #m*l
    
    for i in range(m): 
        z2 = numpy.dot(a1[i],Theta1.T) #1*(n+1) * (n+1)*h =1*h
        a2 = sigmoid(z2)        
        a2 = numpy.append(1, a2)    #1*(h+1)
        
        z3 = numpy.dot(a2, Theta2.T)    #1*(h+1) * (h+1)*l = 1*l
        a3 = sigmoid(z3)
#        if y[i] == 10.0:
#            y[i] = 0.0
        binary_y[i,int(y[i])] = 1
        J += numpy.dot(binary_y[i],numpy.log(a3).T)+numpy.dot((1-binary_y[i]),numpy.log(1-a3).T)
        
        
        #for calculating gradient:
        delta3 = a3 - binary_y[i]   #1*l
        delta2 = numpy.dot(delta3,Theta2[:,1:])*sigmoid(z2)*(1-sigmoid(z2))  #1*l * l*h=1*h
        Theta2_grad += numpy.dot(delta3[numpy.newaxis,:].T,a2[numpy.newaxis,:]) # l*1 * 1*(h+1)= l*(h+1)
        Theta1_grad += numpy.dot(delta2[numpy.newaxis,:].T, a1[i,numpy.newaxis])  # h*1 * 1*(n+1) = h*(n+1)
    J = (-1/m)*J
    
    #part 2: for regularization
    no_bias_theta1 = Theta1[:,1:]   #h*m
    no_bias_theta2 = Theta2[:,1:]   #l*h
    temp_theta1 = (no_bias_theta1 ** 2).sum()
    temp_theta2 = (no_bias_theta2 ** 2).sum()
    
    J_regularized_part = lda/(2*m)*(temp_theta1+temp_theta2)
    J += J_regularized_part
    
    #part3: gradient descent
    Theta1_grad = (1/m)*Theta1_grad
    Theta2_grad = (1/m)*Theta2_grad
    
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (lda/m)*Theta1_grad[:,1:]
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + (lda/m)*Theta2_grad[:,1:]
    
    grad = numpy.append(Theta1_grad.flatten(),Theta2_grad.flatten())
    return (J, grad)


def random_initialize(L_in, L_out):
    """
    randomly initialize the parameter thetas for symmetric breaking
    This neural network just has three layers,therefore accept two params:
    Input: 
        L_in: integer, number of incoming connections
        L_out: integer, number of outgoing connections
    output: a matrix(numpy ndarray) with size (L_out, L_in + 1)
    
    """    
    espilon_init = 0.12
    w = numpy.random.rand(L_out, L_in + 1)*2*espilon_init - espilon_init
    
    return w
    

def predict(Theta1, Theta2, X):
    """
    This function predicts the label of the input X given a trained neural
    network (Theta1, Theta2)
    """
    #total number of training set
    m = X.shape[0]
    
    h1 = numpy.dot(numpy.append(numpy.ones((m,1)), X, axis=1), Theta1.T)  #m*(n=1) * (n+1)*h = m*h
    h1 = sigmoid(h1)
    h2 = numpy.dot(numpy.append(numpy.ones((m,1)), h1, axis=1), Theta2.T) #m*(h+1) * (h+1)*l = m*l
    h2 = sigmoid(h2)
    
    p = numpy.nanargmax(h2, axis=1)  #1*m
    return p