# -*- coding: utf-8 -*-
"""
Created on Tue Apr 03 08:08:48 2018

@author: Dario
"""

import numpy as np
from WordFrequency import WordFrequencyDist_D
import nltk

def preprocessing(raw_sentences, mode, words_to_idx, word_dict):

    def build_vocabulary(raw_sentences, num_words):
        words = []
        for scent in raw_sentences:
            word_scent = nltk.word_tokenize(scent.lower())
            words.extend(word_scent)
        word_dist = WordFrequencyDist_D(words)
        # <unk>, <pad>, <eos>, <bos> are not words, therfore only 19996
        word_dict = {word_dist[i][0]: word_dist[i][1] for i in range(0, num_words)}
        return word_dict, word_dist
        
    def Tokenize(raw_sentences, word_dict, makedummies):
        """
        Takes raw scentences and the word dictinary as input and returns the 
        scentences with new formating:
            - max 30 words per scentence
            - <bos> and <eos> added at beginning and end of each scentence
            - word replaced with <unk> if not in dictionary
            - scentences paded with <pad> if shorter than 30 words
        """
        words_final = []
        tokenized = []
        batchsize = 64
        for scent in raw_sentences:
            word_scent = nltk.word_tokenize(scent.lower())
            #word_scent = scent.split()
            
            for i in range(0,len(word_scent)):
                if not(word_scent[i] in word_dict):
                    word_scent[i] = '<unk>'
                    
            sentence = ['<bos>']
            if len(word_scent) <= 28:
                sentence.extend(word_scent)
                sentence.extend(['<eos>'])
                sentence.extend(['<pad>']*(28-len(word_scent)))
                
                tokenized.append(sentence)
                words_final.extend(sentence)
        if makedummies:
            nr_of_dummy_scentences = batchsize -  len(tokenized)%batchsize
            print('len tok ', len(tokenized))
            print('nr dummy scent ', nr_of_dummy_scentences)
            for _ in range(0,nr_of_dummy_scentences):
                tokenized.append(['<pad>']*30)
                
        return tokenized, words_final

         
    if mode=='training':
        print('preprocessing for training data')
        
        num_words = 19996
        
        # build up the dictionary
        word_dict, total_words = build_vocabulary(raw_sentences=raw_sentences,
                                                  num_words = num_words)   
        
        new_sentences, words_final = Tokenize(raw_sentences = raw_sentences, 
                                               word_dict = word_dict, 
                                               makedummies=True)
        
        word_dist_final = WordFrequencyDist_D(words_final)
        
        print(len(word_dist_final))
        words_to_idx = {word_dist_final[indx][0]: indx for indx in range(0, num_words+4)}
        
        ## Create the training data
        X = [[words_to_idx[w] for w in sent[:-1]] for sent in new_sentences]
        y = [[words_to_idx[w] for w in sent[1:]] for sent in new_sentences]


   
    
    elif mode=='test':
        print('preprocessing for test data')
        # process the test data
        print(len(raw_sentences))
        new_sentences, words_final = Tokenize(raw_sentences, word_dict, makedummies=True)
        
        X = [[words_to_idx[w] for w in sent[:-1]] for sent in new_sentences]
        y = [[words_to_idx[w] for w in sent[1:]] for sent in new_sentences]
        
    elif mode=='1.2':
        print('preprocessing for task 1.2')
        # process the test data
        new_sentences, words_final = Tokenize(raw_sentences, word_dict, makedummies=False)
        
        X = [[words_to_idx[w] for w in sent[:-1]] for sent in new_sentences]
        y = [[words_to_idx[w] for w in sent[1:]] for sent in new_sentences]
    
    return X,y, words_to_idx, word_dict


    