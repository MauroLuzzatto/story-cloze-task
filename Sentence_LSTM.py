# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:34:47 2018

@author: thomas, mauro, dario
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import matplotlib.pyplot as plt
import json

from read_sentences import read_sentences

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score    
from sklearn.metrics import classification_report


#from load_embedding import load_embedding
      
tf.reset_default_graph()   
        
        


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
       
        # initialize the placeholders
        self.input_x = tf.placeholder(shape=[None, self.number_of_sentences, self.embedding_size], dtype=tf.float32) # [batch_size, sentence_length]
        self.input_y = tf.placeholder(shape=[None, self.number_of_sentences, self.embedding_size], dtype=tf.float32)
        

    def build_graph(self, is_training):
        
        batch_size = tf.shape(self.input_x)[0] # make the batch_size dynamic
                
        with tf.variable_scope('rnn_cell', reuse = self.reuseVar):
            
            rnn_cells = []
            for i in range(self.num_layers):
                rnn_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_size, 
                                                         activation = tf.nn.relu))

            if self.num_layers > 1:
                # Stack multiple cells.
                lstm = tf.nn.rnn_cell.MultiRNNCell(cells=rnn_cells, state_is_tuple=True)
            else:
                lstm = rnn_cells[0]
            
#            # Initial state of the LSTM memory
#            lstm = tf.contrib.rnn.LSTMCell(self.lstm_size, 
#                                           forget_bias=0.0, 
#                                           state_is_tuple=True, 
#                                           activation=tf.nn.relu)
        
        with tf.variable_scope('rnn_operations', reuse = self.reuseVar):
            # rnn operation
            print('input x', self.input_x.get_shape())

            self.output, state = tf.nn.dynamic_rnn(cell = lstm, 
                                                  inputs = self.input_x,
                                                  dtype = tf.float32)
            
            self.output = tf.reshape(self.output, [-1, self.lstm_size])
        print('shape output ', self.output.get_shape())
        
        
        with tf.variable_scope('predictions', reuse = self.reuseVar):
            # added activation function
            # self.predictions = tf.nn.relu(tf.add(tf.matmul(self.output, W),b))
            
            dropout_layer = tf.layers.dropout(inputs=self.output, 
                                              rate=self.drop_out_rate, 
                                              training=is_training)
            
            logits = tf.layers.dense(inputs=dropout_layer, 
                                      units=self.embedding_size,
                                      use_bias = True,
                                      kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                      bias_initializer = tf.contrib.layers.xavier_initializer())
            
            print('logits', logits.get_shape())

            
            self.predictions = tf.reshape(logits, [batch_size, self.number_of_sentences, self.embedding_size])
            self.predictions = tf.nn.l2_normalize(self.predictions, axis = 2)
                        
                
        with tf.variable_scope('loss', reuse = self.reuseVar):
            self.input_y = tf.nn.l2_normalize(self.input_y, axis = 2)
            print('input y', self.input_y.get_shape())
            
            # Note that the function assumes that predictions and labels are already unit-normalized.
            # 0: distance between vectors is small, more equal
            # 1: distance between vectros is large, unequal
            
            # only the last sentence is considred to calculate the loss
            self.cosine_distance = tf.losses.cosine_distance(labels=self.input_y[:,-1,:], 
                                                              predictions=self.predictions[:,-1,:], 
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
                                                   decay_steps=5000,
                                                   decay_rate=0.97,
                                                   staircase=False)
        
        
        if is_training:
            with tf.variable_scope('training_operations', reuse = self.reuseVar):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cosine_distance, tvars), self.clip_gradient)
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
                
        
        print('cosine_distance', self.cosine_distance.get_shape())

        summary_learning_rate = tf.summary.scalar('learning_rate', self.learning_rate)
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
            if batch_i%100 == 0:
                print('Training: batch: ', batch_i, '/', num_batches)
                print('cosine_distance: ', cosine_distance, cosine_distance.shape)
                #print('cosine_similarity', 1-cosine_distance)
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
        #print('cosine_similarity', 1-cosine_distance)
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

        return np.array(sentence_prediction), np.array(cosine_distance_right).reshape(1871,1), np.array(cosine_distance_wrong).reshape(1871,1)
                
    
    
    def classification(self, X_true, X_false):
        
        print('start classification')
        
        X = np.concatenate((X_true, X_false), axis = 0)
        y = np.concatenate((np.ones((1871,1)), np.zeros((1871,1))),axis = 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 4, shuffle = True)    


        pipe_lr = Pipeline([('clf', LogisticRegression(random_state=4, penalty='l2', fit_intercept=True))])
        param_grid = [{'clf__C': [0.001, 0.001,0.01, 0.1, 1, 10, 100,1000]}]
        
        gridsearch = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='accuracy', cv=10)
        gridsearch = gridsearch.fit(X_train, y_train.ravel())
        
        print("")
        print("Logistic regression: 10-fold cross-validation")
        print("Best score: ", gridsearch.best_score_)
        print("Best C:", gridsearch.best_params_)
        
        scores = cross_val_score(gridsearch, X_test, y_test.ravel(), scoring='accuracy', cv=10)
        print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 
        
        y_true, y_pred = y_test, gridsearch.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
                

    
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
        
        
    train, valid = read_sentences(pathToData)
    print('sentences loaded')
    valid_stories = get_stories(valid, 'valid', embedding=True)
    print('validation stories made')
    train_stories = get_stories(train,'train', embedding=True)
    print('training stories made')

    return train_stories, valid_stories


def generate_XY(stories):
    """ generate X and y set """

    X = stories[:,0:4,:]
    y = stories[:,1:5,:]
    
    return X, y

#def main(): 
       
   
# define the rnn with LSTM cell
rnn_settings = {
    'number_of_sentences' : 5,        
    'batch_size' : 8,  # change to 8 
    'embedding_size' : 512, #100, 
    'lstm_size' : 512,
    'learning_rate' : 0.001, 
    'number_of_epochs' : 2,
    'clip_gradient' : 10.0,
    'num_layers': 3,
    'dropout_rate': 0,
    'save settings name': 'LSTM3layer2epoch512batchsize8clip10',
    'Training_mode': True
    }


# set paths
pathToData = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test'
pathToLog = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test\log'


# embed stories
save_valid_name='pre_calc_emb_LSTM'

try:
    valid_stories = np.load(pathToData+save_valid_name+'_valid.npy')
    train_stories = np.load(pathToData+save_valid_name+'_train.npy')
    print('loaded pre calculated embeddings')
except:
    print('not able to load precalculated embeddings')
    print('caluculating embedings, this might take a while...')
    train_stories, valid_stories = perform_sentence_embedding(pathToData,embedding=True)
    np.save(pathToData+save_valid_name+'_valid', valid_stories)
    np.save(pathToData+save_valid_name+'_train', train_stories)


X, y = generate_XY(train_stories)

X_train = X[0:80000,:,:]
y_train = y[0:80000,:,:]
        
X_test = X[80000:,:,:]
y_test = X[80000:,:,:]
        


# create rnn graph
RNN = RNN_class(rnn_settings)
# build the graph of the RNN
RNN.build_graph(is_training = True)



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
        saver.save(session, 'graph/' + rnn_settings['save settings name'] + '.ckpt')
        
        with open('graph/' + rnn_settings['save settings name'] + '.json', 'w') as fp:
            json.dump(rnn_settings, fp)
        
        
    else:
        
        #predict the next word word of sentence
        saver = tf.train.Saver()
        saver.restore(session, 'graph/' + rnn_settings['save settings name'] + '.ckpt')
        print('model restored!')
        # generate scentences
        sentence_prediction, cd_right, cd_wrong = RNN.generate_embedded_sentence(session, valid_stories)
        
        plt.scatter(cd_right, cd_wrong)
        count = 0
        total = len(cd_right)
        for ii in range(len(cd_right)):
            if cd_right[ii] < cd_wrong[ii]: # smaller means more equal with prediction
                count += 1
            elif cd_right[ii] == cd_wrong[ii]:
                total -= 1
        
        print('\nscore cosine distance', float(count)/total)
        
        # TODO:
        # train RNN better, learning rate, CRU (?), 
        # check which examples are very wrong regarding cosine distance
        # add further features, theta, sentence embedding
        
        
        """ additional features: best score = 0.6299 """
        X_true = np.concatenate([valid_stories[:,4,:], cd_right, np.arccos(1- cd_right)], axis =1)
        X_false = np.concatenate([valid_stories[:,5,:], cd_wrong, np.arccos(1- cd_wrong)], axis =1)
        
        
        """ Baseline model: best score = 0.626 """
#        X_true = valid_stories[:,4,:]
#        X_false = valid_stories[:,5,:]
        
        
        RNN.classification(X_true, X_false)
            
        
#
#if __name__ == "__main__":
#    # reset the built graph
#    tf.reset_default_graph()
#    # run main method
#    main()            
#    
#    
#    
    
    
    
    
    