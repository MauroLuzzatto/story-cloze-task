# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:34:47 2018

@author: thomas, mauro, dario
"""

import tensorflow as tf
import numpy as np
import os, sys
import json

from feature_extraction import perform_sentence_embedding, generate_XY, \
                                perform_PCA, createFeatures, test_train_split
                                
from recurrent_neural_network import RNN_class
from feature_extraction import PCA_on_features
      

def main(baselineModel, rnn_settings, pca_settings): 
        
    # set paths
    pathToModel = os.path.dirname(os.path.realpath(sys.argv[0]))
    pathToData =  os.path.join(pathToModel, 'data')
    pathToLog = os.path.join(pathToModel, 'log')
    pathToGraph = os.path.join(pathToModel, 'graph')
    
    # embed stories
    save_valid_name='pre_calc_emb_LSTM'
        
    # set the data file names
    fileName = {'train': 'train_stories.csv', 
                'test': 'test_nlu18_utf-8.csv',
                'valid': 'cloze_test_val__spring2016_cloze_test_ALL_val.csv', 
                'valid_2': 'cloze_test_spring2016-test.csv'}
    
    # load embeddings
    try:
        valid_stories = np.load(os.path.join(pathToData, save_valid_name+'_valid.npy'))
        valid_stories_2 = np.load(os.path.join(pathToData, save_valid_name+'_valid_2.npy'))
        train_stories = np.load(os.path.join(pathToData, save_valid_name+'_train.npy'))
        test_stories = np.load(os.path.join(pathToData, save_valid_name+'_test.npy'))
        print('loaded pre calculated embeddings')
    except:
        print('not able to load precalculated embeddings')
        print('caluculating embedings, this might take a while...')
        train_stories, valid_stories, valid_stories_2, \
        test_stories = perform_sentence_embedding(pathToData = pathToData, 
                                                  embedding=True,
                                                  fileName = fileName)
        np.save(os.path.join(pathToData, save_valid_name+'_valid'), valid_stories)
        np.save(os.path.join(pathToData, save_valid_name + '_valid_2'), valid_stories_2)
        np.save(os.path.join(pathToData, save_valid_name+'_train'), train_stories)
        np.save(os.path.join(pathToData, save_valid_name+'_test'), test_stories)
    
    if rnn_settings['pca']:
        # perform PCA on the embedded sentences
        train_stories = perform_PCA(train_stories, rnn_settings['embedding_size'])
        valid_stories = perform_PCA(valid_stories, rnn_settings['embedding_size'])
    
    X_sentence_embedding, y_sentence_embedding = generate_XY(train_stories)
    X_train, X_test, y_train, y_test = test_train_split(X_sentence_embedding, y_sentence_embedding, 80000)
    
    # create rnn graph
    RNN = RNN_class(rnn_settings)
    # build the graph of the RNN
    RNN.build_graph(is_training = True)
    number_of_paramters = RNN.get_num_parameters()
    
    print('Number of Model Paramters:', number_of_paramters)
    print('Number of Model Paramters in Millions:', number_of_paramters/10**6)
    
    # Launch the graph
    with tf.Session() as session:
    
        if rnn_settings['Training_mode']:
            # create rnn graph for the test case
            RNN_validate = RNN_class(rnn_settings)
            # build the graph of the RNN for the test case
            RNN_validate.reuseVar=True
            RNN_validate.build_graph(is_training = False)
            
            saver = tf.train.Saver()
            # Initialize the variables 
            session.run(tf.global_variables_initializer())
            # train the model
            
            writer_train = tf.summary.FileWriter(os.path.join(pathToLog, 'train'))
            writer_train.add_graph(session.graph)
            
            writer_validate = tf.summary.FileWriter(os.path.join(pathToLog, 'validate'))
            writer_validate.add_graph(session.graph)
            
            global_step = 0
            
            # iterate over all epochs
            for epoch_i in range(RNN.number_of_epochs):       
                
                 # shuffle the training data
                 X_train, y_train = RNN.shuffleData(epoch_i, X_train, y_train)
                 print('#epoch: ', epoch_i) 
                 # train the RNN
                 global_step = RNN.train(session, X_train, y_train, writer_train, global_step)
                 # validate the RNN
                 RNN_validate.validate(session, X_test, y_test, writer_validate, global_step)
    
            # export the trained meta-graph
            saver.save(session, os.path.join(pathToGraph, rnn_settings['save settings name'] + '.ckpt'))
            with open(os.path.join(pathToGraph, rnn_settings['save settings name'] + '.json'), 'w') as fp:
                json.dump(rnn_settings, fp)
            
        else:
            #predict the embedding of sentence
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(pathToGraph, rnn_settings['save settings name'] + '.ckpt'))
            print('model restored!')
            
            if baselineModel:
                X_true = valid_stories[:,4,:]
                X_false = valid_stories[:,5,:]
                
                X_valid_true = valid_stories_2[:,4,:]
                X_valid_false = valid_stories_2[:,5,:]
                
                X_test_sent_1 = test_stories[:,4,:]
                X_test_sent_2 = test_stories[:,5,:]
            
            else:
                X_true, X_false = createFeatures(mode = 'valid', 
                                              stories = valid_stories, 
                                              session = session, 
                                              pathToData = pathToData, 
                                              pathToModel = pathToModel, 
                                              RNN = RNN,
                                              fileName = fileName)
                
                X_valid_true, X_valid_false = createFeatures(mode = 'valid_2', 
                                              stories = valid_stories_2, 
                                              session = session, 
                                              pathToData = pathToData, 
                                              pathToModel = pathToModel, 
                                              RNN = RNN,
                                              fileName = fileName)
                
                X_test_sent_1, X_test_sent_2 = createFeatures(mode = 'test', 
                                              stories = test_stories, 
                                              session = session, 
                                              pathToData = pathToData, 
                                              pathToModel = pathToModel, 
                                              RNN = RNN,
                                              fileName = fileName)
            
            # perform principal component analysis on features
            if pca_settings['enable_pca']:
                n_comp=pca_settings['num_components']
                X_true, X_false, pca_mod = PCA_on_features(X_true, X_false, n_comp, False)
                X_valid_true, X_valid_false, _ = PCA_on_features(X_valid_true, X_valid_false, n_comp, pca_mod)
                X_test_sent_1, X_test_sent_2, _ = PCA_on_features(X_test_sent_1, X_test_sent_2, n_comp, pca_mod)
#                X_true, X_false, X_valid_true, X_valid_false, X_test_sent_1,
#                X_test_sent_2 = PCA_on_features(X_true, X_false, X_valid_true,
#                                                X_valid_false, X_test_sent_1,
#                                                X_test_sent_2, n_comp)
            
            RNN.classification(X_true, X_false, X_valid_true, X_valid_false)
        
            RNN.createSubmissionFile(X_test_sent_1, X_test_sent_2)
                
            
if __name__ == "__main__":
    # reset the built graph
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    
    baseline = False
       
    # define the rnn with LSTM cell
    rnn_settings = {
        'number_of_sentences' : 5,        
        'batch_size' : 32,  
        'embedding_size' : 512, 
        'lstm_size' : 64, #64,
        'learning_rate' : 0.0001, # 0.001
        'number_of_epochs' : 80, #, 8,
        'clip_gradient' : 10.0,
        'num_layers': 1,
        'dropout_rate': 0.0,
        'decay_step': 40000,  # use less epochs, overfitting! (lstm 16 was better)
        'save settings name': 'LSTM1layer80epochLSTM64_Emb512batchsize32clip10_dropout0lr0.001', # 'MSE_LSTM3layer7epochLSTM512_Emb512batchsize8clip10_dropout0lr0.001',
        'Training_mode': False, 
        'pca': False
        }
    
    pca_settings = {'enable_pca': False,
                    'num_components': 512}
    # run main method
    main(baseline, rnn_settings, pca_settings)              
        
    