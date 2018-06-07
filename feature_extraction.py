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

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from read_sentences import read_sentences, read_test_sentences
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

from sklearn.preprocessing import normalize

                
def perform_sentence_embedding(pathToData, embedding=True):
    #pathToData = r'/Users/Dario/Desktop/ETH/Freiwillig/NLU/Project2/data/'
    
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
       
    train, valid = read_sentences(pathToData = pathToData)
    print('sentences loaded')
    valid_stories = get_stories(valid, 'valid', embedding)
    print('validation stories made')
    train_stories = get_stories(train,'train', embedding)
    print('training stories made')
    
    test = read_test_sentences(pathToData = pathToData)
    test_stories = get_stories(test,'valid', embedding)
    print('test stories made')
    
    return train_stories, valid_stories, test_stories


def generate_XY(stories):
    """ generate X and y set """
    X = stories[:,0:4,:]
    y = stories[:,1:5,:]
    return X, y


def perform_PCA(data, _n_components):
    print('PCA')
    samples, num_sentence, embedding = data.shape
    pca = PCA(n_components= _n_components, copy = True, whiten = False)
    data = pca.fit_transform(data.reshape(-1,512))
    return data.reshape(-1,num_sentence,_n_components)


#def createSentimentAnalysis(data):
#
##    train, valid = read_sentences(pathToData)
#    data = np.array(data).reshape(-1,6)
#    num_samples, num_sentences = data.shape
#    
#    sid = SentimentIntensityAnalyzer()
#    
#    features_right = []
#    features_wrong = []
#    sent_len_right = []
#    sent_len_wrong = []
#    
#    
#    for sample in range(num_samples):
#        sentence_right = 4
#        sentence_wrong = 5
#        for final_sentence in [sentence_right, sentence_wrong]:
#            
#            if final_sentence == sentence_right:
#                sent_len_right.append(len(data[sample,final_sentence]))                
#                for ii in list(data[sample, 0:4]):
#                    ss = sid.polarity_scores(ii)
#                    features_right.append(ss['compound'])                
#                ss = sid.polarity_scores(data[sample,final_sentence])
#                features_right.append(ss['compound'])
#
#            elif final_sentence == sentence_wrong:
#                sent_len_wrong.append(len(data[sample,final_sentence]))
#                
#                for ii in list(data[sample, 0:4]):
#                    ss = sid.polarity_scores(ii)
#                    features_wrong.append(ss['compound'])                
#                ss = sid.polarity_scores(data[sample,final_sentence])
#                features_wrong.append(ss['compound'])
#            
#    return features_right, features_wrong, sent_len_right, sent_len_wrong
 
    
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
        os.system('word2vecExample.py')
    
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
    print('done context ------------------------------------------------------------------------------------------')
    print('miss ratio: ', context_miss_ratio)
    #add vectors of words in endings
    true_emb, trueSent_miss_ratio = calc_avg_word_emb(stories_text, [4])
    print('done true sentences -----------------------------------------------------------------------------------')
    print('miss ratio: ', trueSent_miss_ratio)
    false_emb, falseSent_miss_ratio = calc_avg_word_emb(stories_text, [5])
    print('done false sentences ----------------------------------------------------------------------------------')
    print('miss ratio: ', falseSent_miss_ratio)
        
    # generate X and y
    X_t = context_emb - true_emb
    y_t = np.ones(len(X_t))
    X_f = context_emb - false_emb
    y_f = np.zeros(len(X_f))
    
    #X = np.concatenate((X_t,X_f))
    #y = np.concatenate((y_t, y_f))        
    
    return X_t, X_f   


def euclidean_distance(x,y):
    """ return euclidean distance between two lists """
    return np.sqrt(np.sum(np.power(x-y,2), axis = 1)).reshape(-1,1)
 
    
def manhattan_distance(x,y):
    """ return manhattan distance between two lists """
    return np.sum(abs(x-y), axis = 1).reshape(-1,1)


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



def createFeatures(mode, valid_stories, session, pathToData, pathToModel, RNN):
    
    num_samples, num_sentences, embedding_size = valid_stories.shape
        
    # generate sentence embeddings
    sentence_prediction, cosine_distance_right, cosine_distance_wrong = RNN.generate_embedded_sentence(session, valid_stories)
    
    #create features for the classification
    valid_true = normalize(valid_stories[:,4,:], norm='l2', axis=1, copy=True)
    valid_false = normalize(valid_stories[:,5,:], norm='l2', axis=1, copy=True)
    sentence_predicted = normalize(sentence_prediction[:,-1,:], norm='l2', axis=1, copy=True)
    
    if mode == 'valid':
        calculateCosineDistanceScore(cosine_distance_right, cosine_distance_wrong)    
        valid_stories_text = perform_sentence_embedding(pathToData, embedding=False)[1]
#        features_right, features_wrong, sent_len_right, sent_len_wrong = createSentimentAnalysis(data = valid_stories)
    
    elif mode == 'test':
        valid_stories_text = perform_sentence_embedding(pathToData, embedding=False)[2]
#        features_right, features_wrong, sent_len_right, sent_len_wrong = createSentimentAnalysis(data = valid_stories)
        
        
    X_right_word_embedding, X_wrong_word_embedding = create_features_word_embeddings_word2vec(valid_stories_text, pathToModel)
    
    
    #sentiment_feature_right = np.mean(np.array(features_right).reshape(-1, 5)[:,0:4],axis = 1) -np.array(features_wrong).reshape(-1, 5)[:,-1]
    #sentiment_feature_wrong = np.mean(np.array(features_wrong).reshape(-1, 5)[:,0:4],axis = 1) -np.array(features_wrong).reshape(-1, 5)[:,-1]


    # concatenate all the features for the classification task        
    X_true = np.concatenate([X_right_word_embedding,
                             #sentiment_feature_right.reshape(num_samples,1),
                             #np.array(features_right).reshape(num_samples, -1),
                             #np.array(sent_len_right).reshape(num_samples, 1),
                             valid_true, 
                             cosine_distance_right, 
                             np.arccos(1- cosine_distance_right), 
                             np.mean(abs(sentence_predicted- valid_true),axis = 1).reshape(num_samples,1),
                             np.var(sentence_predicted- valid_true, axis = 1).reshape(num_samples,1),
                             euclidean_distance(sentence_predicted,valid_true),
                             manhattan_distance(sentence_predicted,valid_true)], axis =1)
    
    
    X_false = np.concatenate([X_wrong_word_embedding,
#                              sentiment_feature_wrong.reshape(num_samples,1),
#                              np.array(features_wrong).reshape(num_samples, -1),
#                              np.array(sent_len_wrong).reshape(num_samples, 1),
                              valid_false, 
                              cosine_distance_wrong, 
                              np.arccos(1- cosine_distance_wrong), 
                              np.mean(abs(sentence_predicted- valid_false),axis = 1).reshape(num_samples,1),
                              np.var(sentence_predicted- valid_false, axis = 1).reshape(num_samples,1),
                              euclidean_distance(sentence_predicted,valid_false),
                              manhattan_distance(sentence_predicted,valid_false)], axis =1)

    return X_true, X_false