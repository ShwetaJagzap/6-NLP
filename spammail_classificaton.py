# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:30:57 2023

@author: HP
"""

import pandas as pd
import numpy as np
# read the csv
df = pd.read_csv("C:/recommendation system/spam.csv.xls")
# check first 10 records
df.head()
#total no of spam and ham 
df.Category.value_counts()
# create one or more co; comprises 0 & 1
#  name of col is spam
df['spam']= df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
df.shape
#----------------------------------------------------------------
# Train test split8from sklearn.model_selection import train_test_split
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.Message, df.spam , test_size=0.2)
# here text_size=0.2 means for testing we use 20% of data 
# and traning we use 80% of data 
# let us check the shpe of x train data and x_test data
x_train.shape
x_test.shape
# let us check the type of x_train and y_train
type(x_train)
type(y_train)
#----------------------------------------------------------------
# create bag of words representation using CounterVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
x_train_cv = v.fit_transform(x_train.values)
x_train_cv
# after creation of bow let us check the shape 
x_train_cv.shape
#----------------------------------------------------------------
# train the naive bayes model
from sklearn.naive_bayes import MultinomialNB
# initialise the model 
model = MultinomialNB()
# train the model 
model.fit(x_train_cv,y_train)
#----------------------------------------------------------------
# create the bag of words representarion using CountVectorizer of x_test
x_train_cv = v.transform(x_test)
#----------------------------------------------------------------
# evaluate performance
from sklearn.metrics import classification_report
y_pred = model.predict(x_train_cv)
print(classification_report(y_test,y_pred))
#----------------------------------------------------------------








