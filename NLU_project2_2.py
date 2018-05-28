'''
NLU project 1

authors: Dario Kneubühler, Mauro Luzzatto, Thomas Brunschwiler
group: 12

'''

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time
import nltk


from load_embedding import load_embedding
from PreprocessingProject2 import preprocessing
from read_sentences import read_sentences


class RNN_class(object):
    """ main class that builds the RNN with an LSTM cell """
    def __init__(self, rnn_settings):
        self.sentence_length = rnn_settings['sentence_length']    # every word corresponds to a time step
        self.embedding_size = rnn_settings['embedding_size']
        self.lstm_size = rnn_settings['lstm_size']
        self.vocabulary_size = rnn_settings['vocabulary_size']
        self.learning_rate = rnn_settings['learning_rate']
        self.number_of_epochs = rnn_settings['number_of_epochs']
        self.clip_gradient = rnn_settings['clip_gradient']
        self.training_mode = rnn_settings['Training_mode'] 
       
        reuseVar=False
       
        # initialize the placeholders
        self.input_x = tf.placeholder(shape=[None, self.sentence_length], dtype=tf.int32) # [batch_size, sentence_length]
        self.input_y = tf.placeholder(shape=[None, self.sentence_length], dtype=tf.int64)
        self.batch_size = tf.shape(self.input_x)[0] # make the batch_size dynamic
        
        with tf.variable_scope("embedding", reuse = reuseVar):
            # create embeddings
            embedding_matrix= tf.get_variable(name="embedding", initializer = tf.random_uniform([self.vocabulary_size, self.embedding_size], -0.1, 0.1))
            embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, self.input_x) # [None, sentence_length, vocab_size, embedding_size]
            
        with tf.variable_scope('rnn_cell', reuse = reuseVar):
            # Initial state of the LSTM memory
            lstm = tf.contrib.rnn.LSTMCell(self.lstm_size, forget_bias=0.0, state_is_tuple=True)
        
        with tf.variable_scope('rnn_operations', reuse = reuseVar):
            # rnn operation
            self.initial_state = lstm.zero_state(self.batch_size, dtype = tf.float32)
            state = self.initial_state
                   
#            
#            inputs = tf.unstack(embedded_inputs, num=self.sentence_length, axis=1)
#            
#            # TODO: change to static
#            self.output, state = tf.nn.static_rnn(cell = lstm, 
#                                                  inputs = inputs,
#                                                  initial_state = self.initial_state,
#                                                  dtype = tf.float32)

            outputs = []
            for word in range(self.sentence_length):
                if word > 0: 
                     tf.get_variable_scope().reuse_variables()
                (lstm_output, state) = lstm(embedded_inputs[:, word], state)
                outputs.append(lstm_output)
            self.output = tf.reshape(tf.concat(outputs, 1), [-1, self.lstm_size])
        
        
   
        with tf.variable_scope('softmax_variables'):
            # Set model weights
            W = tf.get_variable(name = "W", shape=[self.lstm_size, self.vocabulary_size], 
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable(name = "b", shape=[self.vocabulary_size], 
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
         
            
        with tf.variable_scope('logits'):
            logits = tf.add(tf.matmul(self.output, W),b) 
            logits = tf.reshape(logits, [self.batch_size, self.sentence_length, self.vocabulary_size])
            self.sentence_probability = tf.nn.softmax(logits)
                
        with tf.variable_scope('loss'):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.input_y)
            self.loss =  tf.reduce_mean(self.loss, 1)
        
        with tf.variable_scope('predictions'):
            self.predictions = tf.argmax(logits,2)  # [batch_size, sentence_length]
            self.predictions = tf.cast(self.predictions, tf.int64)
            correct = tf.equal(self.predictions, self.input_y)
        
        with tf.variable_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
            tf.summary.scalar('accuracy', self.accuracy)
        
        if self.training_mode:
            with tf.variable_scope('training_operations'):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.clip_gradient)
                optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
                
        
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.summary.merge_all()
        
#        self.distanceSentence = self.calculateSentenceDistance(sentence = embedded_inputs)
#        print(self.distanceSentence)     
    
          # TODO: calculate distance of given sentence and generated sentence
          # TODO: calculate the loss based on the distance of these sentences
        
    
#    def calculateSentenceDistance(self, sentence):
#        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
#        # give words not numbers
#        embeddings = embed(sentence)
#        return embeddings
#
#    def createWordSentence(self, sentence, words_to_idx):
#        keys=list(words_to_idx.keys())
#        values=list(words_to_idx.values())
          # Achtung sentence has several sententences
#        return [[keys[values.index(idx)] for idx in sent] for sent in sentence] 
    
    def test(self, model, session, X, y, task, rnn_settings, words_to_idx, pathToLog):
        """Runs the model on the test data"""
                
        writer = tf.summary.FileWriter(pathToLog)
        writer.add_graph(session.graph)
        
        batch_size = 6#rnn_settings['batch_size']
        self.num_batches = int(len(y)/batch_size)
        iters = 0
        costs = 0
        start_time = time.time()
        
        right, wrong, generated = [],[],[]
        
        for batch_i in range(self.num_batches):
            # get batches
            start = batch_i * batch_size
            end = (batch_i + 1) * batch_size -2
            
            feed_dict = {self.input_x: X[start:end],
                         self.input_y: y[start:end]}
                
            loss, acc, pred, sentence_probability, summary = session.run([self.loss, self.accuracy, 
                                                                self.predictions, self.sentence_probability,
                                                                self.merged], 
                                                                feed_dict)
        
            
            sentence_generated = self.generate_words_greedily(X = X[start:end], 
                                                              words_to_idx = words_to_idx, 
                                                              session = session)
            
            
            #sentence_generated = self.createWordSentence(sentence_generated, words_to_idx)
            sentence_right = X[(batch_i + 1) * batch_size -2]
            sentence_wrong = X[(batch_i + 1) * batch_size -1]
            if batch_i in [0,1,2,3]:
                print('true', sentence_right)
                print('false', sentence_wrong)
                print('generated', sentence_generated)
            
            right.append(sentence_right)
            wrong.append(sentence_wrong)
            generated.append(sentence_generated)
            
            costs += loss
            iters += self.sentence_length
            perplexity = np.exp(costs / iters)
            
            print('batch_i', batch_i, 'num_batches: ', self.num_batches)
            print('Test set: loss: ', np.sum(loss), 'accuracy: ', acc,'perplexity: ', perplexity)
            
            elapsed_time = (time.time() - start_time)/60.
            print('elapsed time (min)', round(elapsed_time,1))
            
            writer.add_summary(summary, batch_i)
            writer.flush()
        
        writer.close()
        
        
        return right, wrong, generated
      
       
        
    def train(self, model, session, X, y, rnn_settings, words_to_idx, pathToLog):
        """Runs the model on the given data."""
        
        writer = tf.summary.FileWriter(pathToLog)
        writer.add_graph(session.graph)
        
        batch_size = rnn_settings['batch_size']
        num_batches = int(len(y)/batch_size)
        len_y = len(y)
        costs = 0
        iters = 0
        start_time = time.time()
        
        # iterate over all epochs
        for epoch_i in range(self.number_of_epochs):
             print('#epoch: ', epoch_i)              
             X, y = self.shuffleData(epoch_i, X,y)
             

             # iterate over all batches
             for batch_i in range(num_batches):
                # get batches
                start = batch_i * batch_size
                end = min((batch_i + 1) * batch_size, len_y)
                
                feed_dict = {self.input_x: X[start:end],
                             self.input_y: y[start:end]}
                    
                _, loss, acc, pred, summary, \
                sentence_probability = session.run([self.train_op, self.loss, self.accuracy, 
                                                    self.predictions, self.merged, self.sentence_probability], 
                                                   feed_dict)
                
                costs += loss
                iters += self.sentence_length
                perplexity = np.exp(costs / iters)
                writer.add_summary(summary, epoch_i)
                writer.flush()
                
                
            
                if batch_i%50 == 0:
                    print('Training: batch: ', batch_i, ' num Batches: ', num_batches)
                    print('loss: ', np.sum(loss), 'accuracy: ', acc, 'perplexity: ', np.mean(perplexity))
                    elapsed_time = (time.time() - start_time)/60.
                    print('elapsed time (min)', round(elapsed_time,1))

                
                
        
        writer.close()
        
    def shuffleData(self, epoch, X, y):
         print('Shuffle data')
         np.random.seed(epoch)
         np.random.shuffle(X)
         np.random.seed(epoch)
         np.random.shuffle(y)
         return X,y

    def cleanOutput(self, X, word_to_idx):
        #create new dict for idx to word operation
        idx_to_word = {y:x for x,y in word_to_idx.items()}
        cleanX=[]
        
        for i in range(len(X)):
            print([idx_to_word[j] for j in X[i]])
            cleanScent=[]
            j = 1
            while True:
                cleanScent.append(idx_to_word[X[i][j]])
                if idx_to_word[X[i][j]]=='<eos>' or j==28: break
                j+=1
            cleanX.append(' '.join(cleanScent))
        return cleanX


    # TODO: remove session dependency
    def generate_words_greedily(self, X, words_to_idx, session):
        """predict the next word of sentence, given the previous words
            load the trained model for this task 1.2"""
        
        #X_original_clean = self.cleanOutput(X, words_to_idx)
        
        for i in range(len(X)): #iterate over all scentences
            # start dimension, predict word by word
            dim = 1
            while True:    
                #compute predictions
                feed_dict = {self.input_x: np.array(X[i]).reshape((1,self.sentence_length)),
                             self.input_y: np.array(X[i]).reshape((1,self.sentence_length))} # input_y is not needed
                                        
                prediction, sentence_probability = session.run([self.predictions, 
                                                                self.sentence_probability], 
                                                                feed_dict)
                #print('prediction', prediction.shape)
                lastpred = prediction[0][dim]
                #print('lastpred', lastpred)
                #print('dim', dim)
                X[i][dim]=lastpred
                dim += 1
                
                if lastpred == words_to_idx['<eos>'] or dim==29: 
                    break
                
            if dim < 29:    
                for pad in range(dim, self.sentence_length):
                    # add <pad> to the generated sentence
                    X[i][pad] = words_to_idx['<pad>']
        
        
        return X
        
            #postprocess X
            #X_clean = self.cleanOutput(X, words_to_idx)
#            self.create_submission_file(X_original_clean, task='originalX')
#            self.create_submission_file(X_clean, task='continuation')


#def main(): 
            
tf.reset_default_graph()

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)


# define the rnn with LSTM cell
rnn_settings = {
    'sentence_length' : 30-1,        
    'batch_size' : 5, # 5 sentences per story
    'embedding_size' : 100, 
    'lstm_size' : 512,     # 1024 for Task C
    'vocabulary_size' : 20000,
    'learning_rate' : 0.001, # default
    'number_of_epochs' : 2,
    'clip_gradient' : 5.0,
    'Training_mode': False
    }

pathMain = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test'
pathData = os.path.join(pathMain, 'data')
pathGraph = os.path.join(pathMain, 'graph')
pathToLog = os.path.join(pathMain, 'log')

#"""
## set paths    
#pathToEmbedding = os.path.join(r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project\data','wordembeddings-dim100.word2vec')
#pathToData = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test'
#train, valid = read_sentences(pathToData) 
#
## TODO: preprocess the data such that all the sentences are equally long
## feed data to network: batch of 1 sentence, batch of 2 sentences, bacht of 3 sentences,...
#
## TODO: start simple, training set, just learn the sentences, 
## create final sentence
#
## TODO: evaluation set, two lists one with the 4 stories, one with the first true, second the folse sentence
#
#
#if rnn_settings['Training_mode']:
#    #proprocess the train, validation and sentence continuation data
#    train_X, train_Y, words_to_idx, \
#    word_dict = preprocessing(raw_sentences = train, 
#                              mode = 'training', 
#                              words_to_idx = None, 
#                              word_dict = None)
#
#else:
#    #proprocess the train, validation and sentence continuation data
#    train_X, train_Y, words_to_idx, \
#    word_dict = preprocessing(raw_sentences = train, 
#                              mode = 'training', 
#                              words_to_idx = None, 
#                              word_dict = None)
#    
#    
#    eval_X, eval_Y, words_to_idx, \
#    word_dict = preprocessing(raw_sentences = valid, 
#                              mode = 'test', 
#                              words_to_idx = words_to_idx, 
#                              word_dict = word_dict)
#
#
#
#
## create rnn graph
#RNN = RNN_class(rnn_settings)
#
## Launch the graph
#with tf.Session() as session:
#
#    if rnn_settings['Training_mode']:    
#        
#        saver = tf.train.Saver()
#        # Initialize the variables 
#        session.run(tf.global_variables_initializer())
#        
#        # load embeddings
#        embedding_matrix= tf.get_variable(name="embedding", \
#                          initializer = tf.random_uniform([rnn_settings['vocabulary_size'],
#                                                           rnn_settings['embedding_size']], -0.1, 0.1))
#
#        load_embedding(session = session, 
#                       vocab = words_to_idx, 
#                       emb = embedding_matrix, 
#                       path = pathToEmbedding,
#                       dim_embedding = rnn_settings['embedding_size'],
#                       vocab_size = rnn_settings['vocabulary_size'])
#        
#        # train the model
#        RNN.train(RNN, session, train_X, train_Y, rnn_settings, words_to_idx, pathToLog)
#        # export the trained meta-graph
#        saver.save(session, os.path.join(pathGraph, 'modelB_new.ckpt'))
#        print('training is finished!')
#        
#
#        
#    else: # test mode
#        
#        saver = tf.train.Saver()
#        saver.restore(session, os.path.join(pathGraph, 'modelB.ckpt')) # should be changed to modelB_new
#        print('model restored')
#        task = 'B'
#        right, wrong, generated = RNN.test(RNN, session, eval_X, eval_Y, task, rnn_settings, words_to_idx, pathToLog)
#        
#        
#"""




#LogisticRegressionWithSentenceLength(eval_X)

def createWordSentence(sentence, words_to_idx, generated):
    keys=list(words_to_idx.keys())
    values=list(words_to_idx.values())
    #Achtung sentence has several sententences
    word_sentence = []
    count = 0
    if not generated:
        for sent in sentence:
            print(count)
            count += 1
            try:
                index_eos = np.argwhere(np.array(sent)==words_to_idx['<eos>'])[0][0] # 2 is eos but would be better using the dict
                index_bos = np.argwhere(np.array(sent)==words_to_idx['<bos>'])[0][0] + 1
                print(index_eos, index_bos)
                word_sentence.append(' '.join([keys[values.index(idx)] for idx in sent[index_bos:index_eos]]))
            except IndexError:
                # remove sentences with only <pad>
                pass
    else:
        
        for sent in sentence:
            four_sentences = []
            for num_sent in sent:
                print(count, num_sent)
                count += 1
                try:
                    index_eos = np.argwhere(np.array(num_sent)==words_to_idx['<eos>'])[0][0] # 2 is eos but would be better using the dict
                    index_bos = np.argwhere(np.array(num_sent)==words_to_idx['<bos>'])[0][0] + 1
                    print(index_eos, index_bos)
                    four_sentences.append(' '.join([keys[values.index(idx)] for idx in num_sent[index_bos:index_eos]]))
                except IndexError:
                    # remove sentences with only <pad>
                    pass
            if four_sentences:
                word_sentence.append(four_sentences)
        
           
     
    return word_sentence





#wrong_sentence = createWordSentence(sentence = wrong, words_to_idx = words_to_idx, generated = False)
#right_sentence = createWordSentence(sentence = right, words_to_idx = words_to_idx, generated = False)
generated_sentence = createWordSentence(sentence = generated, words_to_idx = words_to_idx, generated = True)

     
#if __name__ == '__main__':
#    # reset the built graph
#    tf.reset_default_graph()
#    # run main method
#    main()            

def LogisticRegressionWithSentenceLength(eval_X):
    eval_X = []
    
    for sent in valid:
        eval_X.append(nltk.word_tokenize(sent.lower()))

    batch_size = 6
    X = np.zeros((int(len(eval_X)/batch_size)*2))
    y = np.zeros((int(len(eval_X)/batch_size)*2))
    count = 0
    for batch_i in range(int(len(eval_X)/batch_size)):
        y[count] = 1
        X[count] = len(eval_X[(batch_i + 1) * batch_size -2]) # True
        y[count + 1] = 0
        X[count + 1] = len(eval_X[(batch_i + 1) * batch_size - 1]) # False
        count += 2
    
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 4, shuffle = True)    
    
    for C in [0.001, 0.001,0.01, 0.1, 1, 10, 100,1000]:
        logreg = LogisticRegression(C=C)
        logreg.fit(X_train, y_train) 
        ##        
        ##Make predictions using the testing set
        y_pred = logreg.predict(X_test)
        score = logreg.score(X_test, y_test)
        
        print('score', score)
