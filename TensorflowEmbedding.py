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

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score    

from sklearn.metrics import classification_report
# Reduce logging output.
# tf.logging.set_verbosity(tf.logging.ERROR)
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer


pathToData = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test'
train, valid = read_sentences(pathToData) 


sentences = valid
sid = SentimentIntensityAnalyzer()
for sentence in sentences:
     print(sentence)
     ss = sid.polarity_scores(sentence)
     for k in sorted(ss):
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()



method = 'scipy'
crossValidation = True
withContext = False



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
    

print('load embedding')

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/1')

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  right_emb = session.run(embed(right_sentence))
  wrong_emb = session.run(embed(wrong_sentence))
  context_emb = session.run(embed(context_sentence))




if withContext:
    X_true = np.concatenate((np.array(right_emb).reshape(-1,512), 
                             np.array(right_emb).reshape(-1,512),
                             np.array(right_emb).reshape(-1,512),
                             np.array(right_emb).reshape(-1,512)), axis = 1)

    X_false = np.concatenate((np.array(wrong_emb).reshape(-1,512),
                              np.array(wrong_emb).reshape(-1,512),
                              np.array(wrong_emb).reshape(-1,512),
                              np.array(wrong_emb).reshape(-1,512)), axis = 1)
    
    context_emb = context_emb.reshape(1871,-1)
    X = np.concatenate((X_true - context_emb, X_false- context_emb), axis = 0)

    
else:
    
    X_true = np.array(right_emb).reshape(-1,512)                       
    X_false = np.array(wrong_emb).reshape(-1,512)
    X = np.concatenate((X_true, X_false), axis = 0)


y = np.concatenate((np.ones((1871,1)), np.zeros((1871,1))),axis = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 4, shuffle = True)    

for C in [0.01, 0.1, 1, 10]:
    logreg = LogisticRegression(C=C, penalty = 'l2', fit_intercept = True, random_state = 4)
    logreg.fit(X_train, y_train) 
    y_pred = logreg.predict(X_test)
    score = logreg.score(X_test, y_test)
    print('C', C)
    print('LG score', score)
    print(confusion_matrix(y_test, y_pred))


if crossValidation:
    pipe_lr = Pipeline([('clf', LogisticRegression(random_state=4, penalty='l2', fit_intercept=True))])
    param_grid = [{'clf__C': [0.1, 0.2, 0.5, 0.7, 1]}]
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
    


print('baseline finished')


#for _c in [0.001, 0.001,0.01, 0.1, 1, 10, 100,1000]:
#    clf = svm.SVC(C = _c)
#    clf.fit(X_train, y_train) 
#    y_pred = clf.predict(X_test)
#    score = clf.score(X_test, y_test)
#    print('C', _c)
#    print('SVM score', score)
#    print(confusion_matrix(y_test, y_pred))



if method == 'scipy':
    result_right = np.zeros((1871, 1))
    result_wrong = np.zeros((1871, 1))
    
    from scipy import spatial
    for ii in range(1871):
        result_right[ii] = 1 - spatial.distance.cosine(context_emb[ii], X_true[ii])
        result_wrong[ii] = 1 - spatial.distance.cosine(context_emb[ii], X_false[ii])

elif method == 'sklearn': 
    from sklearn.metrics.pairwise import cosine_distances

    result_right = cosine_distances(context_emb, X_true) 
    result_wrong = cosine_distances(context_emb, X_false)


X = np.concatenate((result_right, result_wrong), axis = 0)
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
