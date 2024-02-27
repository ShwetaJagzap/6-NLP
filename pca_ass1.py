# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:15:42 2023

@author: HP
"""

##bag of words
# this BoW converts unstructured data to structured form
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
corpus=['At least seven isean pharma companies are working to develop vaccine against the corona virus','The deadly virus that has  already infected more than 14 million globally','Bharat biotech is the among the domastic pharma firm the working on the corona virus vaccine in India']
bag_of_word_model=CountVectorizer()
print(bag_of_word_model.fit_transform(corpus).todense())

bag_of_word_df=pd.DataFrame(bag_of_word_model.fit_transform(corpus).todense())
#this will create datframe
bag_of_word_df.columns=sorted(bag_of_word_model.vocabulary_)
bag_of_word_df.head() 
#output
'''14  against  already  among  are  at  ...  that  the  to  vaccine  virus  working
0   0        1        0      0    1   1  ...     0    1   1        1      1        1
1   1        0        1      0    0   0  ...     1    1   0        0      1        0
2   0        0        0      1    0   0  ...     0    4   0        1      1        1'''
#####################################################################
 #bag of words model small
bag_of_word_model_small=CountVectorizer(max_features=5)
bag_of_word_df_small=pd.DataFrame(bag_of_word_model_small.fit_transform(corpus).todense())
bag_of_word_df_small.columns=sorted(bag_of_word_model_small.vocabulary_)
bag_of_word_df.head()
