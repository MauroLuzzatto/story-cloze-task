# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:17:01 2018

@author: thomas, mauro, dario
"""
#%matplotlib qt

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier

from read_sentences import read_sentences

from wrd2vecExample import word2vecEmbeddingCalculation

tf.reset_default_graph()

def save_model(pathname, model):
    pickle.dump(model, open(pathname, 'wb'))
    
def load_model(pathname):
    return pickle.load(open(pathname, 'rb'))

def perform_sentence_embedding(pathToData,embedding=True):
        
    
    def get_stories(sentences, mode, embedding=embedding):
        if mode=='test' or mode=='train':
            nr_sentences = 5
        elif mode=='valid':
            nr_sentences = 6
        else:
            print('please specify mode')
            return
        if embedding:
            # Import the Universal Sentence Encoder's TF Hub module
            embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/1')
            print('sentence encoder importet')
            # Reduce logging output.
            tf.logging.set_verbosity(tf.logging.ERROR)
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                sentences = session.run(embed(sentences))
            stories = np.reshape(sentences,(-1, nr_sentences, 512))
        else:
            stories = np.asarray(sentences)
            stories = np.reshape(stories,(-1, nr_sentences))
            
        return stories

    train, valid = read_sentences(pathToData)
    print('sentences loaded')
    train_stories = get_stories(train,'train', embedding)
    valid_stories = get_stories(valid, 'valid', embedding)
    print('stories made / embedded')

    return valid_stories, train_stories


def generate_X_y_concat(stories):
    X = []
    y = []
    
    for i in range(len(stories)):
        X.append(stories[i,0:5,:])
        y.append(1)
        temp = stories[i,5,:]
        X.append(np.concatenate((stories[i,0:4,:],temp[np.newaxis,:])))
        y.append(0)
        
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y

def generate_X_y_onlylast(stories):
    X = []
    y = []
    for i in range(len(stories)):
        X.append(stories[i,4,:])
        y.append(1)
#        X.append(stories[i,5,:])
#        y.append(0)
    for i in range(len(stories)):
        X.append(stories[i,5,:])
        y.append(0)
        
    return X, y

def generate_X_y_dist_sent_emb(stories, metric):
    """ 
    look at difference beteween sum or context embeddings and true/false ending
    """
    context_embb = np.zeros((len(stories), 512))
    trueSent_embb = np.zeros((len(stories), 512))
    falseSent_embb = np.zeros((len(stories), 512))
    
    for i in range(len(stories)):
        context_embb[i] = np.mean(stories[i,0:4,:], axis=0)
        trueSent_embb[i] = stories[i,4,:]
        falseSent_embb[i] = stories[i,5,:]
    
    # generate X and y
    if metric == 'sub':
        X_t = context_embb - trueSent_embb
        y_t = np.ones(len(X_t))
        X_f = context_embb - falseSent_embb
        y_f = np.zeros(len(X_f))
    elif metric == 'cosine':
        X_t = cosine_similarity(context_embb, trueSent_embb)
        y_t = np.ones(len(X_t))
        X_f = cosine_similarity(context_embb, falseSent_embb)
        y_f = np.zeros(len(X_f))
    else:
        print('please specifi metric: sub or cosine')
        return
    
    X = np.concatenate((X_t,X_f))
    y = np.concatenate((y_t, y_f)) 
    
    return X, y


def generate_X_y_cosine(stories):
    # dont use this it often results in accuracy 1 but we dont know why
    # very strange
    
    context = []
    end_t = []
    end_f =[]
    y = []
    
    for i in range(len(stories)):
        context.append(np.reshape(stories[i,0:4,:],-1))
        end_t.append(np.reshape(np.tile(stories[i,4,:],(4,1)),-1))
        end_f.append(np.reshape(np.tile(stories[i,5,:], (4,1)),-1))
    
    context=np.ones(np.shape(context))#for testing, turned out we can set context to 1 and it doesnt influence the result!!!
#    end_t = np.ones(np.shape(end_t))
#    end_f = np.ones(np.shape(end_f))
    
    X_t = cosine_similarity(context, end_t)
    y_t = np.ones(len(X_t))
    X_f = cosine_similarity(context, end_f)
    y_f = np.zeros(len(X_f))
    
#    X_t = end_t
#    X_f = end_f
    
    X = np.concatenate((X_t,X_f))
    y = np.concatenate((y_t, y_f))
    
    return X, y

def generate_X_cosine(stories):
    context = []
    end = []
    
    for i in range(len(stories)):
        context.append(np.reshape(stories[i,0:4,:],-1))
        end.append(np.reshape(np.tile(stories[i,4,:],(4,1)),-1))
      
    X = cosine_similarity(context, end)
    
    return X

def generate_X_y_cosine_last_story(stories):
    context = []
    end_t = []
    end_f =[]
    y = []
    
    for i in range(len(stories)):
        context.append(np.reshape(stories[i,3,:],-1))
        end_t.append(np.reshape(np.tile(stories[i,4,:],(1,1)),-1))
        end_f.append(np.reshape(np.tile(stories[i,5,:], (1,1)),-1))
      
    X_t = cosine_similarity(context, end_t)
    y_t = np.ones(len(X_t))
    X_f = cosine_similarity(context, end_f)
    y_f = np.zeros(len(X_f))
    
    X = np.concatenate((X_t,X_f))
    y = np.concatenate((y_t, y_f))
    
    return X, y

def predict_on_train(stories, pathToData, modelname):
    nr = 10 # only look at first 10
    st = stories[:nr]
    
    model = load_model(pathToData+modelname)
    X = generate_X_cosine(st)
    pca = PCA(n_components=10, copy=True, whiten=False)
    X = pca.fit_transform(X)
    y_hat = model.predict(X)
    print('y_hat: ', y_hat)
    
    #also display actual sentences
    train, _ = read_sentences(pathToData)
    tr = train[0:(5*nr)]
    tr = np.reshape(tr,(-1, 5))
    for i in range(len(tr)):
        print('sentence ', i)
        print(tr[i])
        print('predicted ', y_hat[i])


def X_y_word_Embeddings_word2vec(stories_text):
    #import pre trained model
    path_to_model = r'C:\Users\Dario\Documents\GitHub\Project-2---Story-Cloze-Test'
    pathSave = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project2_Story_Cloze_Test\data'
    
    model_name = 'final_emb_word2vec.npy'
    dict_name = 'final_emb_dictionary.npy'
    #rev_dict_name = 'final_emb_reverse_dictionary.npy'
    try:
        word_embedding = np.load(os.path.join(path_to_model, model_name))
        word2int = np.load(os.path.join(path_to_model, dict_name)).item()
        print('pretrained word embedding loaded')
    except:
        #calculate embedding
        word2vecEmbeddingCalculation()
    
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
                        print(word.lower(), ' is not in the vocabulary')
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
    
    X = np.concatenate((X_t,X_f))
    y = np.concatenate((y_t, y_f))        
    
    return X, y


pathToData = r'/Users/Dario/Desktop/ETH/Freiwillig/NLU/Project2/data/'
save_valid_name='pre_calc_valid_emb'
save_train_name='pre_calc_train_emb'
try:
    valid_stories = np.load(pathToData+save_valid_name+'.npy')
    train_stories = np.load(pathToData+save_train_name+'.npy')
    valid_stories_text = np.load(pathToData+save_valid_name+'_text'+'.npy')
    train_stories_text = np.load(pathToData+save_train_name+'_text'+'.npy')
    print('loaded pre calculated sentence embeddings and sentences')
except:
    print('not able to load precalculated sentence embeddings')
    print('caluculating embeddings, this might take a while...')
    valid_stories_text, train_stories_text = perform_sentence_embedding(pathToData,embedding=False)
    np.save(pathToData+save_valid_name+'_text', valid_stories_text)
    np.save(pathToData+save_train_name+'_text', train_stories_text)
    print('text stories saved')
    valid_stories, train_stories = perform_sentence_embedding(pathToData,embedding=True)
    np.save(pathToData+save_valid_name, valid_stories)
    np.save(pathToData+save_train_name, train_stories)


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#X, y = generate_X_y_concat(valid_stories)
#X1, y = generate_X_y_cosine(valid_stories)
#X, y = generate_X_y_cosine_last_story(valid_stories)

X1, y1 = generate_X_y_onlylast(valid_stories)
X1 = np.asarray(X1)
pca = PCA(n_components=20, copy=True, whiten=False)
X1 = pca.fit_transform(X1)

X3, y3 = generate_X_y_dist_sent_emb(valid_stories,metric='sub') #uses the sentence embedding
X2, y2 = X_y_word_Embeddings_word2vec(valid_stories_text) # uses the word embedding
# use pca to have 100 components in each
pca = PCA(n_components=100, copy=True, whiten=False)
X2 = pca.fit_transform(X2)
X3 = pca.fit_transform(X3)

# using multiple sources of features
Xnew = np.append(X2, X3, axis=1)
ynew = y2

#------------------------------------------
#choose which X and y to use
#------------------------------------------
X=X2
y=y2

X = np.array(X).reshape(len(X),-1)

if not('kind' in locals()): kind='forest' # specify how t do the estimation





if kind=='old':
    #old
    #old
    #old
    
    print('shape X before pca ', np.shape(X))
#    pca = PCA(n_components=20, copy=True, whiten=False)
#    X = pca.fit_transform(X)
  #######  X = np.reshape(X[:,2],(-1,1)) # just look as most relevant feature with n components 3
   ###### X = pca.singular_values_
    print('shape X after pca ', np.shape(X))
    
    #train test split random state influences the score a kind of heavily, this seems kind of strange
    # max and min scores for sweeping random state from 0 to 10
    # diff_sent_emb             0.60(c=0.1) - 0.64(c=1)        also best C is dependable on random state!!!!!
    # only last sent emb        0.58 - 0.63
    # word embb                 0.55 - 0.59
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =8, shuffle = True)   
    
    
    for C in [0.001,0.01, 0.1, 1, 10, 20, 50, 100]:
        logreg = LogisticRegression(C=C, penalty = 'l2', fit_intercept=True)
        logreg.fit(X_train, y_train) 
        ##        
        ##Make predictions using the testing set
        y_pred = logreg.predict(X_test)
        score = logreg.score(X_test, y_test)
        print('C', C)
        print('score', score)
    
    save_model(pathToData+'simple_linear_model', logreg)
    
    #predict_on_train(train_stories, pathToData, 'simple_linear_model')

elif kind=='forest':
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =8, shuffle = True) 
    
    for n_est in [1, 10, 100, 1000]:
        tree = RandomForestClassifier(n_estimators=n_est)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        score = tree.score(X_test, y_test)
        print('n estimators: ', n_est)
        print('score: ', score)


elif kind=='gridsearch':

    logistic = LogisticRegression(penalty='l2', fit_intercept=True)
    svm = svm.SVC(kernel='rbf', degree=3)
    
    pca = PCA(whiten=False)
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
#    pipe = Pipeline(steps=[('pca', pca), ('logistic', svm)])
    
    
    n_components = [10, 100]
    Cs = [0.01, 0.1, 1, 10, 100]
    param_grid = [{'logistic__C': Cs, 'pca__n_components': n_components}]
    
    gs = GridSearchCV(estimator=pipe,
                      param_grid=param_grid,
                      scoring='accuracy', cv=5)
    
    gs = gs.fit(X, y)
 
    
    print("")
    print("Logistic regression: t-fold cross-validation")
    print("Best score: ", gs.best_score_)
    print("Best C:", gs.best_params_)
#    save_model(pathToData+'simple_linear_model_gridsearch', gs)#How to save the best model ?????????????????????????????????????????????????????????????
    
    clf = gs.best_estimator_
    clf.fit(X, y)
    scores = cross_val_score(gs, X, y, scoring='accuracy', cv=10)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 
    
    
 