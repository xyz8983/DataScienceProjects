# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:37:04 2017

@author: sylvia
"""

import pandas as pd
import numpy
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#import seaborn as sns

training = pd.read_csv('./train.csv')
testing = pd.read_csv('./test.csv')
relationship = training.corr()
#sns.heatmap(relationship, vmax=1.0, square = True)

#2. Drop some useless variables.


training_label = training['SalePrice']
training_label = numpy.log1p(training_label)  #transformed into normal distribution
training = training.drop(['SalePrice'], axis = 1) 
all_data = pd.concat((training,testing),axis = 0)


all_data_f_drop = ['GarageYrBlt', 'PoolQC', 'MiscFeature', 'GarageCars','GarageFinish',
                   'GarageCond','GarageQual','BsmtFinType1','BsmtFinType2','MoSold','1stFlrSF']
for f_drop in all_data_f_drop:
    all_data = all_data.drop(f_drop, axis = 1)

#check all missing data, see if there is any column can be deleted
number_missing = all_data.isnull().sum().sort_values(ascending=False)
number_missing = number_missing[number_missing!=0]
percentage_missing = number_missing/all_data.shape[0]
missing_data = pd.concat([number_missing, percentage_missing], axis = 1, keys=['number','percentage'])
missing_data

all_data_f_drop = ['Alley', 'Fence', 'FireplaceQu']
for f_drop in all_data_f_drop:
    all_data = all_data.drop(f_drop, axis = 1)
    
all_data_f_zero = ['MasVnrArea','BsmtFullBath','BsmtFinSF1','TotalBsmtSF',
                  'GarageArea']
for f_zero in all_data_f_zero:                  
    all_data[f_zero] = all_data[f_zero].fillna(0)
all_data_f_other = ['Exterior2nd','SaleType','Exterior1st']
for f_other in all_data_f_other:
    all_data[f_other] = all_data[f_other].fillna('Other')
all_data_f_mean=['LotFrontage','BsmtUnfSF']
for f_mean in all_data_f_mean:
    all_data[f_mean] = all_data[f_mean].fillna(numpy.mean(testing[f_mean]))
all_data_f_frequent = ['MSZoning','KitchenQual','Utilities','Functional']
for f_frequent in all_data_f_frequent:
    all_data[f_frequent] = all_data[f_frequent].fillna(all_data[f_frequent].value_counts().index[0])
na_column = all_data.isnull().sum()
na_column = list(na_column[na_column!=0].index)
for x in na_column:
    all_data[x]=all_data[x].fillna('None')    

#log transform skewed numeric features:
numeric_var = all_data.dtypes[all_data.dtypes != "object"].index
skewed_var = training[numeric_var].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_var = skewed_var[skewed_var > 0.75]
skewed_var = skewed_var.index
all_data[skewed_var] = numpy.log1p(all_data[skewed_var])

all_data = pd.get_dummies(all_data)

scale_f_list = ['LotFrontage','LotArea','MasVnrArea',
'BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','2ndFlrSF',
'GrLivArea','TotRmsAbvGrd','GarageArea','WoodDeckSF','EnclosedPorch','3SsnPorch',
'ScreenPorch','PoolArea']


for feature in scale_f_list:
    all_data[feature] = preprocessing.scale(all_data[feature])

#train linear regression model with regularization
line = training.shape[0]
training = all_data[:line]
testing = all_data[line:]    
training_x, validation_x, training_y, validation_y = train_test_split(training, training_label, test_size=0.3,random_state=10)

error_rate=[]
alphas = [0.01,0.06,0.1,1,1.6,2,5,8,10,12,15,20]
for alpha in alphas:
    linearM = linear_model.Ridge(alpha)
    linearM.fit(training_x, training_y)
    predicted_y = linearM.predict(validation_x)
    error_rate.append(metrics.mean_squared_error(validation_y,predicted_y))
plt.plot(alphas,error_rate)
min_error, idx = min((min_error, idx) for (idx, min_error) in enumerate(error_rate))
print('alpha and smallest error',alphas[idx], min_error)

#predict using the testing set
linearM = linear_model.Ridge(alphas[idx])
linearM.fit(training_x, training_y)

predicted_y_test = linearM.predict(testing)
#since training_label get a log transformation
predicted_y_test = numpy.expm1(predicted_y_test)
testing_y = pd.DataFrame(predicted_y_test, columns = ['SalePrice'])
testing_y.index += 1461
testing_y.index.name = 'Id'
testing_y.to_csv('salePrice_prediction_submission.csv', sep=',')
