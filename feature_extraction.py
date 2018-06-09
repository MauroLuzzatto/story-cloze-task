# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 09:56:50 2018

@author: mauro
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import nltk
import re
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def perform_sentence_embedding(pathToData, embedding=True, fileName = None):
    
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/1')
    print('sentence encoder importet')
    
    def get_stories(sentences, mode, embedding=embedding):
        if mode=='test' or mode=='train':
            nr_sentences = 5
        elif mode=='valid':
            nr_sentences = 6
        else:
            print('please specify mode')
            return
        
        if embedding:
            # Reduce logging output.
            tf.logging.set_verbosity(tf.logging.ERROR)
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                sentences = session.run(embed(sentences))
            print('shape: ', np.shape(sentences))
            stories = np.reshape(sentences, (-1, nr_sentences, 512))
        else:
            stories = np.asarray(sentences)
            stories = np.reshape(stories,(-1, nr_sentences))
            
        return stories
    
    train = read_sentences_train(pathToData = pathToData,   fileName = fileName['train'])
    valid = read_sentences_valid(pathToData = pathToData,   fileName = fileName['valid'])
    valid_2 = read_sentences_valid(pathToData = pathToData, fileName = fileName['valid_2'])
    test = read_sentences_test(pathToData = pathToData,     fileName = fileName['test'])
    
    print('sentences loaded')
    valid_stories = get_stories(valid, 'valid', embedding)
    print('validation stories made')
    valid_stories_2 = get_stories(valid_2, 'valid', embedding)
    print('validation stories made')
    train_stories = get_stories(train,'train', embedding)
    print('training stories made')
    test_stories = get_stories(test, 'valid', embedding)
    print('test stories made')
    
    return train_stories, valid_stories, valid_stories_2, test_stories


def generate_XY(stories):
    """ generate X and y set """
    X = stories[:,0:4,:]
    y = stories[:,1:5,:]
    return X, y


def perform_PCA(data, _n_components):
    #only for rnn sentence embeddings
    print('PCA')
    samples, num_sentence, embedding = data.shape
    pca = PCA(n_components= _n_components, copy = True, whiten = False)
    data = pca.fit_transform(data.reshape(-1,512))
    return data.reshape(-1,num_sentence,_n_components)

    
def create_features_word_embeddings_word2vec(stories_text, path_to_model):
    #import pre trained model
    
    model_name = 'final_emb_word2vec.npy'
    dict_name = 'final_emb_dictionary.npy'
    #rev_dict_name = 'final_emb_reverse_dictionary.npy'
    try:
        word_embedding = np.load(os.path.join(path_to_model, model_name))
        word2int = np.load(os.path.join(path_to_model, dict_name)).item()
        print('pretrained word embedding loaded')
    except:
        #calculate embedding
        os.system('word2vec.py')
    
    def calc_avg_word_emb(stories_text, sentences):
        """
        calculate average embedding of the words in a story
        the sentence vector specifies which sentences of the stories are taken into account
        """
        avg_word_emb = np.zeros((len(stories_text), 128))
        nr_of_words_not_in_voc = 0
        tot_nr_of_words = 0
        for i in range(len(stories_text)):
            temp_sum = np.zeros(128)
            cnt=0 # counts words in a story
            for j in sentences:
                wordList = re.sub("[^\w]", " ",  stories_text[i][j]).split()
                for word in wordList:
                    try:
                        temp_sum = temp_sum + word_embedding[word2int[word.lower()]]
                        cnt += 1
                    except:
                        #print(word.lower(), ' is not in the vocabulary')
                        nr_of_words_not_in_voc+=1
                    tot_nr_of_words+=1
                    
            avg_word_emb[i,:] = temp_sum/cnt
        miss_ratio = nr_of_words_not_in_voc / tot_nr_of_words
        return avg_word_emb, miss_ratio
    
    #add vectors of words in context
    context_emb, context_miss_ratio = calc_avg_word_emb(stories_text, range(0,4))
    print('done context')
    print('miss ratio: ', context_miss_ratio)
    #add vectors of words in endings
    true_emb, trueSent_miss_ratio = calc_avg_word_emb(stories_text, [4])
    print('done true sentences')
    print('miss ratio: ', trueSent_miss_ratio)
    false_emb, falseSent_miss_ratio = calc_avg_word_emb(stories_text, [5])
    print('done false sentences')
    print('miss ratio: ', falseSent_miss_ratio)
        
    # generate X and y
    X_t = context_emb - true_emb
    X_f = context_emb - false_emb
     
    return X_t, X_f   


def euclidean_distance(x,y):
    """ return euclidean distance between two lists """
    return np.sqrt(np.sum(np.power(x-y,2), axis = 1)).reshape(-1,1)
 
    
def manhattan_distance(x,y):
    """ return manhattan distance between two lists """
    return np.sum(abs(x-y), axis = 1).reshape(-1,1)


def test_train_split(X, y, split = 80000):
    X_train = X[0:split,:,:]
    y_train = y[0:split,:,:]
        
    X_test = X[split:,:,:]
    y_test = y[split:,:,:]
    return X_train, X_test, y_train, y_test


def calculateCosineDistanceScore(cosine_distance_right, cosine_distance_wrong):
    plot = False
    count = 0
    total = len(cosine_distance_right)
    for ii in range(len(cosine_distance_right)):
        if cosine_distance_right[ii] < cosine_distance_wrong[ii]: # smaller means more equal with prediction
            count += 1
    if plot:
        plt.scatter(cosine_distance_right, cosine_distance_wrong)
    print('\nscore cosine distance', float(count)/total)


def createFeatures(mode, stories, session, pathToData, pathToModel, RNN, fileName):
    
    num_samples, num_sentences, embedding_size = stories.shape
    # generate sentence embeddings
    sentence_prediction, cosine_distance_1, cosine_distance_2 = RNN.generate_embedded_sentence(session, stories)
    
    #create features for the classification
    sentence_1_embedding = normalize(stories[:,4,:], norm='l2', axis=1, copy=True)
    sentence_2_embedding = normalize(stories[:,5,:], norm='l2', axis=1, copy=True)
    sentence_predicted_embedding = normalize(sentence_prediction[:,-1,:], norm='l2', axis=1, copy=True)
    
    if mode == 'valid':
        #calculateCosineDistanceScore(cosine_distance_right, cosine_distance_wrong)    
        stories_text = perform_sentence_embedding(pathToData, embedding=False, fileName = fileName)[1]
    
    elif mode == 'valid_2':
        #calculateCosineDistanceScore(cosine_distance_right, cosine_distance_wrong)    
        stories_text = perform_sentence_embedding(pathToData, embedding=False, fileName = fileName)[2]
    
    elif mode == 'test':
        stories_text = perform_sentence_embedding(pathToData, embedding=False, fileName = fileName)[3]
        
        
    X_1_word_embedding, X_2_word_embedding = create_features_word_embeddings_word2vec(stories_text, pathToModel)
    
    # concatenate all the features for the classification task        
    X_1 = np.concatenate([
                         sentence_1_embedding, #A1
                         X_1_word_embedding, #B1
                         np.var(sentence_predicted_embedding- sentence_1_embedding, axis = 1).reshape(num_samples,1), #C1
                         (sentence_predicted_embedding- sentence_1_embedding).reshape(num_samples,-1), #C2
                         manhattan_distance(sentence_predicted_embedding, sentence_1_embedding), #C3 
                         euclidean_distance(sentence_predicted_embedding, sentence_1_embedding), # C4
                         cosine_distance_1
                         ], axis =1)
    
    X_2 = np.concatenate([
                          sentence_2_embedding,
                          X_2_word_embedding,
                          np.var(sentence_predicted_embedding- sentence_2_embedding, axis = 1).reshape(num_samples,1),
                          (sentence_predicted_embedding- sentence_2_embedding).reshape(num_samples,-1),
                          manhattan_distance(sentence_predicted_embedding, sentence_2_embedding),
                          euclidean_distance(sentence_predicted_embedding, sentence_2_embedding),
                          cosine_distance_2 
                          ], axis =1)

    return X_1, X_2


def PCA_on_features(X1, X2, n_comp, pca_model):
    #perform PCA on features before the classification
    #print('shape X1 ', np.shape(X1))
    lenX1 = len(X1)
    Xconcat = np.concatenate((X1,X2), axis = 0)
    #print('shape concat ', np.shape(Xconcat))
    if not(pca_model):
        #perform pca
        pca_model = PCA(n_components= n_comp, copy = True, whiten = False)
        pca_model.fit(Xconcat)
        Xconcat = pca_model.transform(Xconcat)
    else:
        Xconcat = pca_model.transform(Xconcat)
    
    X1 = Xconcat[:lenX1]
    X2 = Xconcat[lenX1:2*lenX1]
    
    #print('shape X1 after ', np.shape(X1))
    #print('shape X2 after ', np.shape(X2))
    
    return X1, X2, pca_model


def read_sentences_valid(pathToData, fileName):
    
    validation = pd.read_csv(os.path.join(pathToData, fileName))
    valid = []
    for index in range(validation.shape[0]):
        valid.append(validation['InputSentence1'][index])
        valid.append(validation['InputSentence2'][index])
        valid.append(validation['InputSentence3'][index])
        valid.append(validation['InputSentence4'][index])
        
        if validation['AnswerRightEnding'][index] == 1:
            valid.append(validation['RandomFifthSentenceQuiz1'][index])
            valid.append(validation['RandomFifthSentenceQuiz2'][index])
        
        elif validation['AnswerRightEnding'][index] == 2:
            valid.append(validation['RandomFifthSentenceQuiz2'][index])
            valid.append(validation['RandomFifthSentenceQuiz1'][index])
    return valid


def read_sentences_train(pathToData, fileName):
    
    sentences = pd.read_csv(os.path.join(pathToData, fileName))
    train = []
    for index in range(sentences.shape[0]):
        train.append(sentences['sentence1'][index])
        train.append(sentences['sentence2'][index])
        train.append(sentences['sentence3'][index]) 
        train.append(sentences['sentence4'][index]) 
        train.append(sentences['sentence5'][index])    
    return train


def read_sentences_test(pathToData, fileName):
        
    test_data = pd.read_csv(os.path.join(pathToData, fileName), header = None)
    test = []
    for index in range(test_data.shape[0]):
        test.append(test_data[0][index])
        test.append(test_data[1][index])
        test.append(test_data[2][index]) 
        test.append(test_data[3][index]) 
        test.append(test_data[4][index])
        test.append(test_data[5][index])        
    return test
