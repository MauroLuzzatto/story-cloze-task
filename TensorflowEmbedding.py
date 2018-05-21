# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:17:01 2018

@author: mauro
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from read_sentences import read_sentences
from sklearn import svm

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)



pathToData = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test'
train, valid = read_sentences(pathToData) 

batch_size = 6
y = np.ones((int(len(valid)/batch_size)*2)) * 5
right_sentence = []
wrong_sentence = []
context_sentence = []
count = 0


for batch_i in range(int(len(valid)/batch_size)):
    
    right_sentence.append(valid[(batch_i + 1) * batch_size -2]) # True
    wrong_sentence.append(valid[(batch_i + 1) * batch_size - 1]) # False
    
    start = batch_i * batch_size
    context_sentence.append(valid[start]) 
    context_sentence.append(valid[start+1]) 
    context_sentence.append(valid[start+2]) 
    context_sentence.append(valid[start+3])
    
#    y_ANN[count] = 1
#    y_ANN[count + 1] = 0
#    count += 2
#    X_ANN.append(valid[(batch_i + 1) * batch_size -2]) # True
#    X_ANN.append(valid[(batch_i + 1) * batch_size - 1]) # False
#    



# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/1')

# Compute a representation for each message, showing various lengths supported.
#messages = X_list


with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  
  right_emb = session.run(embed(right_sentence))
  wrong_emb = session.run(embed(wrong_sentence))
  context_emb = session.run(embed(context_sentence))

#  message_embeddings = session.run(embed(messages))     

      
#  for i, message_embedding in enumerate(np.array(right_emb).tolist()):
#    print("Message: {}".format(right_emb[i]))
#    print("Embedding size: {}".format(len(right_emb)))
#    message_embedding_snippet = ", ".join((str(x) for x in right_emb[:3]))
#    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
#    plt.scatter(range(len(right_emb)), right_emb)
#    plt.show()


X_true = np.concatenate((np.array(right_emb).reshape(-1,512), 
                         np.array(right_emb).reshape(-1,512),
                         np.array(right_emb).reshape(-1,512),
                         np.array(right_emb).reshape(-1,512)), axis = 1)

X_false = np.concatenate((np.array(wrong_emb).reshape(-1,512),
                          np.array(wrong_emb).reshape(-1,512),
                          np.array(wrong_emb).reshape(-1,512),
                          np.array(wrong_emb).reshape(-1,512)), axis = 1)

context_emb = context_emb.reshape(1871,-1)

X = np.concatenate((X_true, X_false), axis = 0)
y = np.concatenate((np.ones((1871,1)), np.zeros((1871,1))),axis = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 4, shuffle = True)    

for C in [0.001, 0.001,0.01, 0.1, 1, 10, 100,1000]:
    logreg = LogisticRegression(C=C, penalty = 'l2', fit_intercept = True)
    logreg.fit(X_train, y_train) 
    y_pred = logreg.predict(X_test)
    score = logreg.score(X_test, y_test)
    print('C', C)
    print('LG score', score)
    print(confusion_matrix(y_test, y_pred))


for _c in [0.001, 0.001,0.01, 0.1, 1, 10, 100,1000]:
    clf = svm.SVC(C = _c)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print('C', _c)
    print('SVM score', score)
    print(confusion_matrix(y_test, y_pred))



def neural_network_with_estimator():
    # not used
    
    # Use pre-trained universal sentence encoder to build text vector
    review = hub.text_embedding_column('review', 'https://tfhub.dev/google/universal-sentence-encoder/1')
    
    split = int(len(X_list)*0.75)
    
    features_train = {'review': np.array(X_list[:split])}
    features_test =  {'review': np.array(X_list[split:])}
    labels_train = y.reshape(-1,1)[:split]
    labels_test = y.reshape(-1,1)[split:]
    
    
    training_input_fn = tf.estimator.inputs.numpy_input_fn(features_train, 
                                                           labels_train, 
                                                           shuffle=True, 
                                                           batch_size=8, 
                                                           num_epochs=10)
    
    test_input_fn = tf.estimator.inputs.numpy_input_fn(features_test, 
                                                       labels_test, 
                                                       shuffle=True, 
                                                       batch_size=4, 
                                                       num_epochs=1)
    
    estimator = tf.estimator.DNNClassifier(
                           hidden_units=[500, 100],
                           feature_columns=[review],
                           n_classes=2,
                           optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))
        
    estimator.train(training_input_fn, max_steps=10000)    
    estimator.evaluate(input_fn = test_input_fn)
