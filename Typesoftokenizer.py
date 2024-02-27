# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:19:31 2023

@author: HP
"""

import re
sentence5="sharat twitted ,Whittnessing 70th republic day India from Rajpath,\new Delhi,Memorizing perfoemance by Indian Army"
re.sub(r'([^\s\w]|_)+',' ',sentence5).split()
'''
['sharat',
 'twitted',
 'Whittnessing',
 '70th',
 'republic',
 'day',
 'India',
 'from',
 'Rajpath',
 'ew',
 'Delhi',
 'Memorizing',
 'perfoemance',
 'by',
 'Indian',
 'Army']'''
###########################################################
###extracting n-grams
#n-grams using custom using theree techniques
#1-custome defined function
#2.NLTK
#3.TextBlot
###################################################
#extacting n-grams using custom defined function
import re
def n_gram_extraction(input_str, n):
    tokens=re.sub(r'([^\s\w]|_)+',' ',input_str).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])
n_gram_extraction("The cute little boy is playing with kitten",2)
'''['The', 'cute']
['cute', 'little']
['little', 'boy']
['boy', 'is']
['is', 'playing']
['playing', 'with']
['with', 'kitten']'''
n_gram_extraction("The cute little boy is playing with kitten",3)
'''['The', 'cute', 'little']
['cute', 'little', 'boy']
['little', 'boy', 'is']
['boy', 'is', 'playing']
['is', 'playing', 'with']
['playing', 'with', 'kitten']'''
########################################################################
from nltk import ngrams
#extracting n-grams with nltk
list(ngrams("The cute little boy is playing with kitten".split(),2))
'''[('The', 'cute'),
 ('cute', 'little'),
 ('little', 'boy'),
 ('boy', 'is'),
 ('is', 'playing'),
 ('playing', 'with'),
 ('with', 'kitten')]'''
list(ngrams("The cute little boy is playing with kitten".split(),3))
'''[('The', 'cute', 'little'),
 ('cute', 'little', 'boy'),
 ('little', 'boy', 'is'),
 ('boy', 'is', 'playing'),
 ('is', 'playing', 'with'),
 ('playing', 'with', 'kitten')]'''
#########################################################
#pip install textblob
import textblob
from textblob import TextBlob
blob=TextBlob("the cute littel boy is playing with kitten")
blob.ngrams(n=2)
'''[WordList(['the', 'cute']),
 WordList(['cute', 'littel']),
 WordList(['littel', 'boy']),
 WordList(['boy', 'is']),
 WordList(['is', 'playing']),
 WordList(['playing', 'with']),
 WordList(['with', 'kitten'])]'''
blob.ngrams(n=3)
'''[WordList(['the', 'cute', 'littel']),
 WordList(['cute', 'littel', 'boy']),
 WordList(['littel', 'boy', 'is']),
 WordList(['boy', 'is', 'playing']),
 WordList(['is', 'playing', 'with']),
 WordList(['playing', 'with', 'kitten'])]'''
############################################################

####Tokenization using Keras
#pip install tensorflow
#pip intsall Keras
sentence5
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(sentence5)
'''['sharat',
 'twitted',
 'whittnessing',
 '70th',
 'republic',
 'day',
 'india',
 'from',
 'rajpath',
 'ew',
 'delhi',
 'memorizing',
 'perfoemance',
 'by',
 'indian',
 'army']'''
###########################################################
###teokenization using TextBlob
from textblob import TextBlob
blob=TextBlob(sentence5)
blob.words
'''WordList(['sharat', 'twitted', 'Whittnessing', '70th', 
'republic', 'day', 'India', 'from', 'Rajpath', 'ew', 
'Delhi', 'Memorizing', 'perfoemance', 'by', 'Indian', 'Army'])'''
############################################################
###tweet tokenizer
from nltk.tokenize import TweetTokenizer
tweet_tokenizer=TweetTokenizer()
tweet_tokenizer.tokenize(sentence5)
'''['sharat',
 'twitted',
 ',',
 'Whittnessing',
 '70th',
 'republic',
 'day',
 'India',
 'from',
 'Rajpath',
 ',',
 'ew',
 'Delhi',
 ',',
 'Memorizing',
 'perfoemance',
 'by',
 'Indian',
 'Army']'''
############################################################
##multi word expression
from nltk.tokenize import MWETokenizer
sentence5
mwe_tokenizer=MWETokenizer([('republic','day')])
mwe_tokenizer.tokenize(sentence5.split())
'''['sharat',
 'twitted',
 ',Whittnessing',
 '70th',
 'republic_day',
 'India',
 'from',
 'Rajpath,',
 'ew',
 'Delhi,Memorizing',
 'perfoemance',
 'by',
 'Indian',
 'Army']'''
mwe_tokenizer.tokenize(sentence5.replace('i',' ').split())
'''['sharat',
 'tw',
 'tted',
 ',Wh',
 'ttness',
 'ng',
 '70th',
 'republ',
 'c',
 'day',
 'Ind',
 'a',
 'from',
 'Rajpath,',
 'ew',
 'Delh',
 ',Memor',
 'z',
 'ng',
 'perfoemance',
 'by',
 'Ind',
 'an',
 'Army']
'''
##########Regular Expression tokenizer#################
from nltk.tokenize import RegexpTokenizer
reg_tokenizer=RegexpTokenizer('\w+|/$[\d\.]+|\S+')
reg_tokenizer.tokenize(sentence5)
'''['sharat',
 'twitted',
 ',Whittnessing',
 '70th',
 'republic',
 'day',
 'India',
 'from',
 'Rajpath',
 ',',
 'ew',
 'Delhi',
 ',Memorizing',
 'perfoemance',
 'by',
 'Indian',
 'Army']'''

#########################################################
##whitespace tokenizer
from nltk.tokenize import WhitespaceTokenizer
wh_tokenizer=WhitespaceTokenizer()
wh_tokenizer.tokenize(sentence5)
'''['sharat',
 'twitted',
 ',Whittnessing',
 '70th',
 'republic',
 'day',
 'India',
 'from',
 'Rajpath,',
 'ew',
 'Delhi,Memorizing',
 'perfoemance',
 'by',
 'Indian',
 'Army']'''
##############################################################
from nltk.tokenize import WordPunctTokenizer
wp_tokenizer=WordPunctTokenizer()
wp_tokenizer.tokenize(sentence5)
'''['sharat',
 'twitted',
 ',',
 'Whittnessing',
 '70th',
 'republic',
 'day',
 'India',
 'from',
 'Rajpath',
 ',',
 'ew',
 'Delhi',
 ',',
 'Memorizing',
 'perfoemance',
 'by',
 'Indian',
 'Army']'''
#############################################################
sentence6="I love playing cricket.Criket players practices hard their inning"
from nltk.stem import RegexpStemmer
regex_stemmer=RegexpStemmer('ing$')
' '.join(regex_stemmer.stem(wd)for wd in sentence6.split())
'''I love play cricket.Criket players practices hard their inn'''
#############################################################
sentence7="Before eating, it would be nice to sanitize your hand with a sanitizer"
from nltk.stem.porter import PorterStemmer
ps_stemmer=PorterStemmer()
words=sentence7.split()
" ".join([ps_stemmer.stem(wd) for wd in words])
###############################################################
####lemitization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()
sentence8='The codes executed today are far better than what we execute generally'
words=word_tokenize(sentence8)
" ".join([lemmatizer.lemmatize(word) for word in words])
'''The code executed today are far better than what we execute
 generally'''
############################################################
###singularize and pluaralization
from textblob import TextBlob
sentence9=TextBlob("she sells seashells on the seashore")
words=sentence9.words
###we want to make word [2] i.e.seashells in singular form
sentence9.words[2].singularize()
##we want word 5 i.e seashore in plural form
sentence9.words[5].pluralize()
#'seashores'
########################################################
#language translation from spanish to english
from textblob import TextBlob
en_blob=TextBlob(u'muy bien')
en_blob.translate(from_lang='es',to='en')
#TextBlob("very good")
#es.spanish en:English
##################################################################
#custom stopwords removel
from nltk import word_tokenize
sentence9="She sells seashells on seashore"

custom_stop_word_list=['she','on','the','am','is']
words=word_tokenize(sentence9)
" ".join([word for word in words if word.lower()not in custom_stop_word_list])
'''sells seashells seashore'''
#select words which are not  in defined list
########################################################################
#extracting general features from raw text
#number of words
#detect presens of wh words
#palarity
#subjectivity
#language identification
################################################################
#to identify the number of words
import pandas as pd
df=pd.DataFrame([['The vaccine for covid 19 will be anounced on 1 st august'],['Do you know how much expections the world population in having from this research?'],['The risk of virus will come to an end on 31st july']])
df.columns=['text']
df
######################################################################
#now lwt us measure the number of words
from textblob import TextBlob
df['number_of_words']=df['text'].apply(lambda x:len(TextBlob(x).words))
df['number_of_words']
#output
'''0    12
1    14
2    12
Name: number_of_words, dtype: int64'''
#########################################################################
##detect presence of words wh
wh_words=set(['why','who','which','what','where','when','how'])
df['is_wh_words_present']=df['text'].apply(lambda x:True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
df['is_wh_words_present']
#output
'''0    False
1     True
2    False
Name: is_wh_words_present, dtype: bool'''
#########################################################################
###polarity of the sentence
df['polarity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']
#output
'''0    0.0
1    0.2
2    0.0
Name: polarity, dtype: float64'''
##########################################
sentence10='I like it very much'
pol=TextBlob(sentence10).sentiment.polarity
pol
#o/p: 0.26

sentence10='This is fantastic example and I like it very much'
pol=TextBlob(sentence10).sentiment.polarity
pol
#o/p:0.33

sentence10='This is helpfull example but I would like have prefer another one'
pol=TextBlob(sentence10).sentiment.polarity
pol
#o/p:0.0

sentence10='This was helpful example but I would like have prefer another one'
pol=TextBlob(sentence10).sentiment.polarity
pol
#o/p:0.0

########################################################

###subjectivity of the dataframe df and check whether their is 
df['subjectivity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['subjectivity']
'''Output: 
0    0.0
1    0.2
2    0.0
Name: subjectivity, dtype: float64'''
####################################################################
##to find language of the sentence, this part of code will get http error
df['language']=df['text'].apply(lambda x:TextBlob(str(x)).detect_language())
df['language']=df['text'].apply(lambda x:TextBlob(str(x)).detect_language())
#############################################################################
#
import pandas as pd