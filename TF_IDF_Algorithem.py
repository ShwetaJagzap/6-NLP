# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:51:56 2023

@author: HP
"""

#how to use TF_Idf algorithm
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
corpus=['The mouse has a tiny littel mouse','The cat saw the mouse','The end of the mouse story']
#step initialize count vector 
cv=CountVectorizer()
#to count the total no of TF
word_count_vector=cv.fit_transform(corpus)
word_count_vector.shape
'output:(3, 10)   '
#now next step is to apply IDF
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
#This matrics is the raw matrics form, let us convert in the dataframe
df_idf=pd.DataFrame(tfidf_transformer.idf_,index=cv.get_feature_names_out(),columns=['idf_weights'])
#sort ascending
df_idf.sort_values(by=['idf_weights'])
'''output:
    idf_weights
mouse      1.000000
the        1.000000
cat        1.693147
end        1.693147
has        1.693147
littel     1.693147
of         1.693147
saw        1.693147
story      1.693147
tiny       1.693147'''
######################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
corpus=['Thore eating pizzza,Loki is eating pizza, Ironman ate pizza already',
        'Apple is announcing new iphone tommorow',
        'Tesla is announcing new model-3 tommorow',
        'Google is announcing new pixel-6 tommorow',
        'Microsoft is announcing new surface tommorow',
        'Amazon is announcing new eco_dot tommorow',
        'I am eating biryani and you are eating grapes']
#lets create the vectorizer and fit the corpus and transform them according
v=TfidfVectorizer()
v.fit(corpus)
transform_output=v.transform 
