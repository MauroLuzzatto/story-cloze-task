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

from read_sentences import read_sentences


pathToData = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test'
train, valid = read_sentences(pathToData) 

batch_size = 6
y = np.ones((int(len(valid)/batch_size)*2)) * 5
X_list = []
count = 0
for batch_i in range(int(len(valid)/batch_size)):
    y[count] = 1
    y[count + 1] = 0
    X_list.append(valid[(batch_i + 1) * batch_size -2]) # True
    X_list.append(valid[(batch_i + 1) * batch_size - 1]) # False
    count += 2



# Use pre-trained universal sentence encoder to build text vector
review = hub.text_embedding_column('review', 'https://tfhub.dev/google/universal-sentence-encoder/1')

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/1')

# Compute a representation for each message, showing various lengths supported.
messages = X_list

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed(messages))

#  for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
#    print("Message: {}".format(messages[i]))
#    print("Embedding size: {}".format(len(message_embedding)))
#    message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:3]))
#    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
#    plt.scatter(range(len(message_embedding)), message_embedding)
#    plt.show()



X = np.array(message_embeddings).reshape(-1,512)
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 4, shuffle = True)    

for C in [0.001, 0.001,0.01, 0.1, 1, 10, 100,1000]:
    logreg = LogisticRegression(C=C, penalty = 'l2')
    logreg.fit(X_train, y_train) 
    ##        
    ##Make predictions using the testing set
    y_pred = logreg.predict(X_test)
    score = logreg.score(X_test, y_test)
    print('C', C)
    print('score', score)









split = int(len(X_list)*0.75)

features = {'review': np.array(X_list[:split])}
features2 =  {'review': np.array(X_list[split:])}
labels = y.reshape(-1,1)[:split]
labels2 = y.reshape(-1,1)[split:]



training_input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, 
                                                       shuffle=True, batch_size=8, num_epochs=10)

test_input_fn = tf.estimator.inputs.numpy_input_fn(features2, labels2, 
                                                   shuffle=True, batch_size=4, num_epochs=1)

estimator = tf.estimator.DNNClassifier(
       hidden_units=[500, 100],
       feature_columns=[review],
       n_classes=2,
       optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))



estimator.train(training_input_fn, max_steps=10000)

estimator.evaluate(input_fn = test_input_fn)
# print("Loss is " + str(loss))
