# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:02:42 2018

@author: mauro
"""

import pandas as pd
import os

def read_sentences(pathToData):

    sentences = pd.read_csv(os.path.join(pathToData, 'train_stories.csv'))
    validation =  pd.read_csv(os.path.join(pathToData, 'cloze_test_val__spring2016_cloze_test_ALL_val.csv'))
        
    
    train = []
    for index in range(sentences.shape[0]):
        train.append(sentences['sentence1'][index])
        train.append(sentences['sentence2'][index])
        train.append(sentences['sentence3'][index]) 
        train.append(sentences['sentence4'][index]) 
        train.append(sentences['sentence5'][index])
        
        
    valid = []
    for index in range(validation.shape[0]):
        valid.append(validation['InputSentence1'][index])
        valid.append(validation['InputSentence2'][index])
        valid.append(validation['InputSentence3'][index])
        valid.append(validation['InputSentence4'][index])
        
        if validation['AnswerRightEnding'][index] == 1:
            #print('\nTrue ending:')
            valid.append(validation['RandomFifthSentenceQuiz1'][index])
            #print('\nFalse ending:')
            valid.append(validation['RandomFifthSentenceQuiz2'][index])
        
        elif validation['AnswerRightEnding'][index] == 2:
            #print('\nTrue ending:')
            valid.append(validation['RandomFifthSentenceQuiz2'][index])
            #print('\nFalse ending:')
            valid.append(validation['RandomFifthSentenceQuiz1'][index])
    

    return train, valid


def read_test_sentences(pathToData):

    test_data = pd.read_csv(os.path.join(pathToData, 'test_nlu18_utf-8.csv'), header = None)
    test = []
    for index in range(test_data.shape[0]):
        test.append(test_data[0][index])
        test.append(test_data[1][index])
        test.append(test_data[2][index]) 
        test.append(test_data[3][index]) 
        test.append(test_data[4][index])
        test.append(test_data[5][index])
            
    return test

        
    
  

