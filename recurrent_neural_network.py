# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:01:58 2018

@author: mauro
"""

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score    
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

class RNN_class(object):
    """ main class that builds the RNN with an LSTM cell """
    
    def __init__(self, rnn_settings):
        self.number_of_sentences = rnn_settings['number_of_sentences']-1
        self.embedding_size = rnn_settings['embedding_size']
        self.lstm_size = rnn_settings['lstm_size']
        self.learning_rate = rnn_settings['learning_rate']
        self.number_of_epochs = rnn_settings['number_of_epochs']
        self.clip_gradient = rnn_settings['clip_gradient']
        self.num_layers = rnn_settings['num_layers']
        self.drop_out_rate = rnn_settings['dropout_rate']
        self.batch_size = rnn_settings['batch_size']
        self.reuseVar=False
        self.decay_step = rnn_settings['decay_step']
       
        # initialize the placeholders
        self.input_x = tf.placeholder(shape=[None, self.number_of_sentences, self.embedding_size], dtype=tf.float32) # [batch_size, sentence_length]
        self.input_y = tf.placeholder(shape=[None, self.number_of_sentences, self.embedding_size], dtype=tf.float32)
        

    def build_graph(self, is_training):
        
        # extract the correct batch_size
        batch_size = tf.shape(self.input_x)[0] # make the batch_size dynamic
                
        with tf.variable_scope('rnn_cell', reuse = tf.AUTO_REUSE):
            
            rnn_cells = []
            for i in range(self.num_layers):
                rnn_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_size))

            if self.num_layers > 1:
                # Stack multiple cells.
                lstm = tf.nn.rnn_cell.MultiRNNCell(cells=rnn_cells, state_is_tuple=True)
            else:
                
                lstm = tf.contrib.rnn.BasicLSTMCell(num_units = self.lstm_size)
                    
        with tf.variable_scope('rnn_operations', reuse = tf.AUTO_REUSE):
            # rnn operation

            self.output, state = tf.nn.dynamic_rnn(cell = lstm, 
                                                  inputs = self.input_x,
                                                  dtype = tf.float32)
            
            self.output = tf.reshape(self.output, [-1, self.lstm_size])
        
        
        with tf.variable_scope('predictions', reuse = tf.AUTO_REUSE):
            dropout_layer = tf.layers.dropout(inputs=self.output, 
                                              rate=self.drop_out_rate, 
                                              training=is_training)
            
            predictions = tf.layers.dense(inputs=dropout_layer, 
                                      units=self.embedding_size,
                                      use_bias = True,
                                      kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                      bias_initializer = tf.contrib.layers.xavier_initializer())
                        
            self.predictions = tf.reshape(predictions, [batch_size, self.number_of_sentences, self.embedding_size])
            self.predictions_normalized = tf.nn.l2_normalize(self.predictions, axis = 2)
                        
                
        with tf.variable_scope('cosine_distance_loss', reuse = tf.AUTO_REUSE):
            
            self.input_y_normalized = tf.nn.l2_normalize(self.input_y, axis = 2)
            self.input_y_last_sentence_normalized = self.input_y[:,-1,:]
            
            # Note that the function assumes that predictions and labels are already unit-normalized.
            # 0: distance between vectors is small, more equal
            # 1: distance between vectros is large, unequal
            
            # if self.loss_function =='cosine_distance':
            # only the last sentence is considred to calculate the loss
            self.cosine_distance = tf.losses.cosine_distance(labels=self.input_y_normalized[:,-1,:], 
                                                              predictions=self.predictions_normalized[:,-1,:], 
                                                              axis = 1,
                                                              reduction = 'none')
            
            # 0: no similarity exists between compared vectors 
            # 1: the compared vectors are absolutely similar
            self.cosine_similarity = 1- self.cosine_distance
            self.theta = tf.acos(self.cosine_similarity)
            self.total_loss =  tf.reduce_mean(self.cosine_distance)
                         
        
        global_step = tf.Variable(1, name='global_step', trainable=False)

        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=self.decay_step,
                                                   decay_rate=0.97,
                                                   staircase=False)
        
        if is_training:
            with tf.variable_scope('training_operations', reuse = tf.AUTO_REUSE):
                tvars = tf.trainable_variables()      
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cosine_distance, tvars), self.clip_gradient)       
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
                

        summary_learning_rate = tf.summary.scalar('learning_rate', learning_rate)
        summary_cos_dis = tf.summary.histogram('cosine_distance', self.cosine_distance)
        summary_cos_sim = tf.summary.histogram('cosine_similarity', self.cosine_similarity)
        summary_theta = tf.summary.histogram('theta', self.theta)
        summary_loss = tf.summary.scalar('loss', self.total_loss)
                
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.summary.merge([summary_cos_dis, 
                                        summary_cos_sim, 
                                        summary_theta, 
                                        summary_loss,
                                        summary_learning_rate])
    
#        print('cosine_distance', self.cosine_distance.get_shape())
#        print('input_y_normalized', self.input_y_normalized.get_shape())
#        print('shape output ', self.output.get_shape())
#        print('input x', self.input_x.get_shape())
#        print('predictions', predictions.get_shape())



    
    def get_num_parameters(self):
        """
        return: total number of trainable parameters.
        """
        num_parameters = 0
        # Iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # multiplying dimension values
            num_parameters += local_parameters

        return num_parameters
        
    
    def train(self, session, X, y, writer, global_step):
        """Runs the model on the given data."""
    
        num_batches = int(len(y)/self.batch_size)
        # iterate over all batches
        for batch_i in range(num_batches):
            # get batches
            start = batch_i * self.batch_size
            end = min((batch_i + 1) * self.batch_size, len(y))
            
            feed_dict = {self.input_x: X[start:end],
                         self.input_y: y[start:end]}
            
            _, cosine_distance, pred, summary, loss_total = session.run([self.train_op, self.cosine_distance, 
                                                                         self.predictions, self.merged, 
                                                                         self.total_loss], 
                                                                         feed_dict)
            if batch_i%1000 == 0:
                print('Training: batch: ', batch_i, '/', num_batches)
                print('global step', global_step, '/', num_batches * self.number_of_epochs)
                print('loss total', loss_total)
                writer.add_summary(summary, global_step)
            global_step += 1
        
        return global_step
    
    
    def validate(self, session, X, y, writer, global_step):
        """Runs the model on the given data."""
        feed_dict = {self.input_x: X,
                     self.input_y: y}
        
        cosine_distance, summary, loss_total = session.run([self.cosine_distance, 
                                                            self.merged, 
                                                            self.total_loss], 
                                                            feed_dict)
        
        print('Test error')
        print('cosine_distance: ', cosine_distance, cosine_distance.shape)
        print('Test loss total', loss_total)
        writer.add_summary(summary, global_step)
                    
        
    def shuffleData(self, epoch, X, y):
         print('Shuffle data')
         np.random.seed(epoch)
         np.random.shuffle(X)
         np.random.seed(epoch)
         np.random.shuffle(y)
         return X,y
    
    
    def generate_embedded_sentence(self, session, X):
        """predict the next word of sentence, given the previous words"""
        
        num_samples, num_sentences, embedding_size = X.shape
        
        sentence_prediction = []
        cosine_distance_right = []
        cosine_distance_wrong = []

        for sample in range(num_samples):
            sentence_right = 4
            sentence_wrong = 5
            for final_sentence in [sentence_right, sentence_wrong]:
                
                sentences_X = X[sample, 0:4,:]
                sentences_y = np.concatenate([X[sample, 1:4,:], X[sample,final_sentence,:].reshape(1,embedding_size)], axis = 0)
                
                feed_dict = {self.input_x: sentences_X.reshape(1,num_sentences -2, embedding_size),
                             self.input_y: sentences_y.reshape(1,num_sentences -2, embedding_size)}
            
                cosine_distance, predictions, summary = session.run([self.cosine_distance, 
                                                                     self.predictions, 
                                                                     self.merged], 
                                                                     feed_dict)
                if final_sentence == sentence_right:
                    #print('right: cosine_distance', cosine_distance[:,-1][0][0])
                    cosine_distance_right.append(cosine_distance[0][0])
                elif final_sentence == sentence_wrong:
                    #print('wrong: cosine_distance', cosine_distance[:,-1][0][0])
                    cosine_distance_wrong.append(cosine_distance[0][0])
                
            sentence_prediction.append(predictions[0,:])

        return np.array(sentence_prediction), np.array(cosine_distance_right).reshape(-1,1), np.array(cosine_distance_wrong).reshape(-1,1)
                
    
    def classification(self, X_true, X_false, X_valid_true, X_valid_false):
        # X_true and X_false are features corresponding to the two possible endings in the data set
        # the names might be confunsing, but it emerged from the beginning of the project when we sorted the 
        # sentences in a way such that the first sentence was always true and the second sentence always wrong
        
        self.X_train = np.concatenate((X_true, X_false), axis = 0)
        # class according to the order of the sentences
        self.y_train = np.concatenate((np.ones((X_true.shape[0],1)), 
                                       np.ones((X_false.shape[0],1))*2),
                                       axis = 0)
        
        self.X_test = np.concatenate((X_valid_true, X_valid_false), axis = 0)
        # class according to the order of the sentences
        self.y_test = np.concatenate((np.ones((X_true.shape[0],1)), 
                                      np.ones((X_false.shape[0],1))*2),
                                      axis = 0)
                
        
        print('\nStart classification\n')
        
        pipe = Pipeline([('clf', LogisticRegression(random_state=4, penalty='l2', fit_intercept=True))])
        param_grid = [{'clf__C': [0.001, 0.001,0.01, 0.1, 1, 10, 100, 1000]}]
        
        self.gridsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=10)
        self.gridsearch = self.gridsearch.fit(self.X_train, self.y_train.ravel())
        
        print("Logistic regression: 10-fold cross-validation")
        print("Training score, stories individually calssified: ", self.gridsearch.best_score_)
        print("Best C:", self.gridsearch.best_params_)
        
        print('Test score (old valditation dataset), stories individually calssified:')
        scores = cross_val_score(self.gridsearch, self.X_test, self.y_test.ravel(), scoring='accuracy', cv=5)
        print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 
        
        y_pred = self.gridsearch.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))
        
        self.prediction_classification = self.Prediction(self.gridsearch, X_valid_true, X_valid_false)
        
        score = float(self.prediction_classification.count(1))/len(self.prediction_classification)
        print('Validation score (new validation dataset), stories joint and classified', score)
        

    def createSubmissionFile(self, X_test_sent_1, X_test_sent_2):
        
        fileName = 'output_project2_NLU2018_group_13.csv'
        
        X = np.concatenate((self.X_train, self.X_test), axis = 0)
        # class according to the order of the sentences
        y = np.concatenate((self.y_train, self.y_test),axis = 0)
                
        # train the model on the full dataset
        clf = LogisticRegression(C = self.gridsearch.best_score_, 
                                 random_state=4, 
                                 penalty='l2', 
                                 fit_intercept=True)
        
        clf.fit(X, y.ravel())
        
        final_prediction = self.Prediction(clf, X_test_sent_1, X_test_sent_2)
        
            
        with open(fileName, 'w') as file_handler:
            for item in final_prediction:
                    file_handler.write("{}\n".format(item))
        print('output file created')
        
    
    def Prediction(self, clf, X_test_sent_1, X_test_sent_2):
        
#        print('classes', clf.classes_)
        # predict the probability of each sentence being true
        y_pred_1 = clf.predict_proba(X_test_sent_1)
        y_pred_2 = clf.predict_proba(X_test_sent_2)
        #print('ypred true: ', y_pred_1)

        
        final_prediction = []
        for ii in range(0,len(y_pred_1)):
            if y_pred_1[ii,0] > y_pred_2[ii,0]:
                final_prediction.append(clf.classes_[0])
            elif y_pred_1[ii,0] <= y_pred_2[ii,0]:
                final_prediction.append(clf.classes_[1])
                
        return final_prediction