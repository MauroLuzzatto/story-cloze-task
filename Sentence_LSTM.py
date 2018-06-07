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
                                perform_PCA, createFeatures
                                

from recurrent_neural_network import RNN_class
      
tf.reset_default_graph()   
tf.logging.set_verbosity(tf.logging.ERROR)
  

#def main(): 
       
   
# define the rnn with LSTM cell
rnn_settings = {
    'number_of_sentences' : 5,        
    'batch_size' : 32,  
    'embedding_size' : 512, 
    'lstm_size' : 64, #64,
    'learning_rate' : 0.0001, # 0.001
    'number_of_epochs' : 48, #, 8,
    'clip_gradient' : 10.0,
    'num_layers': 1,
    'dropout_rate': 0.0,
    'decay_step': 10000,  # use less epochs, overfitting! (lstm 16 was better)
    'save settings name': 'LSTM1layer48epochLSTM64_Emb512batchsize32clip10_dropout0lr0.001', # 'MSE_LSTM3layer7epochLSTM512_Emb512batchsize8clip10_dropout0lr0.001',
    'Training_mode': True, 
    'pca': False,
    'loss type': 'cosine_distance'
    }


# set paths
pathToGraph = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test\Project-2---Story-Cloze-Test\graph'
pathToData = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test'
pathToLog = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test\log'
pathToModel = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test\Project-2---Story-Cloze-Test'
# embed stories
save_valid_name='pre_calc_emb_LSTM'
baselineModel = False

               
# load embeddings
try:
    valid_stories = np.load(pathToData+save_valid_name+'_valid.npy')
    train_stories = np.load(pathToData+save_valid_name+'_train.npy')
    test_stories = np.load(pathToData+save_valid_name+'_test.npy')
    print('loaded pre calculated embeddings')
except:
    print('not able to load precalculated embeddings')
    print('caluculating embedings, this might take a while...')
    train_stories, valid_stories, \
    test_stories = perform_sentence_embedding(pathToData = pathToData, 
                                              embedding=True)
    
    np.save(pathToData+save_valid_name+'_valid', valid_stories)
    np.save(pathToData+save_valid_name+'_train', train_stories)
    np.save(pathToData+save_valid_name+'_test', test_stories)


if rnn_settings['pca']:
    # perform PCA on the embedded sentences
    train_stories = perform_PCA(train_stories, rnn_settings['embedding_size'])
    valid_stories = perform_PCA(valid_stories, rnn_settings['embedding_size'])


X_sentence_embedding, y_sentence_embedding = generate_XY(train_stories)

X_train = X_sentence_embedding[0:80000,:,:]
y_train = y_sentence_embedding[0:80000,:,:]
        
X_test = X_sentence_embedding[80000:,:,:]
y_test = y_sentence_embedding[80000:,:,:]
        

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
        
        #predict the next word word of sentence
        saver = tf.train.Saver()
        saver.restore(session, os.path.join(pathToGraph, rnn_settings['save settings name'] + '.ckpt'))
        print('model restored!')
        
        
        if baselineModel:
            X_true = valid_stories[:,4,:]
            X_false = valid_stories[:,5,:]
        
        
        X_true, X_false = createFeatures(mode = 'valid', 
                                         valid_stories = valid_stories, 
                                         session = session, 
                                         pathToData = pathToData, 
                                         pathToModel = pathToModel, 
                                         RNN = RNN)
        
        RNN.classification(X_true, X_false)
        
        X_test_sent_1, X_test_sent_2 = createFeatures(mode = 'test', 
                                                      valid_stories = test_stories, 
                                                      session = session, 
                                                      pathToData = pathToData, 
                                                      pathToModel = pathToModel, 
                                                      RNN = RNN)
        
        RNN.createSubmissionFile(X_test_sent_1, X_test_sent_2)
                
        
#
#if __name__ == "__main__":
#    # reset the built graph
#    tf.reset_default_graph()
#    # run main method
#    main()            
#    
#    
#    
    
    
    
    
    