# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:02:42 2018

@author: mauro
"""

import pandas as pd
import os
import nltk
from WordFrequency import WordFrequencyDist_D

def read_sentences():

    pathToData = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2'
    
    sentences = pd.read_csv(os.path.join(pathToData, 'train_stories.csv'))
    validation =  pd.read_csv(os.path.join(pathToData, 'cloze_test_val__spring2016_cloze_test_ALL_val.csv'))
    
    num_sent_train = sentences.shape[0]
    num_sent_valid = validation.shape[0]
    total_sent = 5 * num_sent_train
    
    def getStory(sentences, index = 0):
        print(sentences['sentence1'][index])
        print(sentences['sentence2'][index])
        print(sentences['sentence3'][index])
        print(sentences['sentence4'][index])
        print(sentences['sentence5'][index])
    
    def getValidationStory(sentences, index = 0):
        print(sentences['InputSentence1'][index])
        print(sentences['InputSentence2'][index])
        print(sentences['InputSentence3'][index])
        print(sentences['InputSentence4'][index])
        
        if sentences['AnswerRightEnding'][index] == 1:
            print('\nTrue ending:')
            print(sentences['RandomFifthSentenceQuiz1'][index])
            print('\nFalse ending:')
            print(sentences['RandomFifthSentenceQuiz2'][index])
        
        elif sentences['AnswerRightEnding'][index] == 2:
            print('\nTrue ending:')
            print(sentences['RandomFifthSentenceQuiz2'][index])
            print('\nFalse ending:')
            print(sentences['RandomFifthSentenceQuiz1'][index])
    
        
    
    #getStory(sentences, 1)
    
    #getValidationStory(validation, 2)
    
    train = []
    for index in range(num_sent_train):
        train.append(sentences['sentence1'][index])
        train.append(sentences['sentence2'][index])
        train.append(sentences['sentence3'][index]) 
        train.append(sentences['sentence4'][index]) 
        train.append(sentences['sentence5'][index])
    

    valid = None
    return train, valid



#
## PoS tagging using Spacy
#import spacy.en
#import os
#from spacy.en import English, LOCAL_DATA_DIR
#
#
#data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
#nlp = English(parser=False, tagger=True, entity=False)
#
#
#def print_fine_pos(token):
#    return (token.tag_)
#
#def pos_tags(sentence):
#    sentence = unicode(sentence, "utf-8")
#    tokens = nlp(sentence)
#    tags = []
#    for tok in tokens:
#        tags.append((tok,print_fine_pos(tok)))
#    return tags
#
#sentenceToTest = validation['InputSentence1'][1]
#print(pos_tags(sentenceToTest))
#
