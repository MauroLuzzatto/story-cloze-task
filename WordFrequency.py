# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 23:42:40 2018

@author: Dario
"""

import numpy as np
import time
#import os
#import codecs
import operator

#"/Users/Dario/Pictures/PondIce.wav
pathData = '/Users/Dario/Desktop/ETH/Freiwillig/NLU/Project/Data'


# load data
#
##def SentencePreProcessing(): 
#filename = os.path.join(pathData, 'sentences.train')
#fp = codecs.open(filename, 'r', 'utf-8')
#
#
#file = open(filename, 'rt')
#text = file.read()
#file.close()

def WordFrequencyDist_veryOld(words):
    t_start = time.time()
    dist = []
    dist.append([words[0], 1])
    for w in words[1:]:
        foundflag = False
        for d in dist:
            if d[0] == w:
                d[1]=d[1]+1
                foundflag = True
                break
        if not(foundflag):
            dist.append([w, 1])
    print('time: '+str(time.time()-t_start))
    #sort and stuff
    dist_ar = np.asarray(dist)
    arg_sort = np.flip(np.argsort(dist_ar[:,1]),axis=0)
    print('time: '+str(time.time()-t_start))
    return dist_ar[arg_sort]
  

          
def WordFrequencyDist_old(words):
    """ takes list of words as input
    returns the frequncy of words
    this newer version is much faster than
    the first one"""
    t_start = time.time()
    dist = {words[0]: 1}
    for w in words[1:]:
        if w in dist:
            dist[w] += 1
        else:
            dist[w] = 1 
    print('time: '+str(time.time()-t_start))
    sorted_dist = np.flip(sorted(dist.items(), key=operator.itemgetter(1)),axis=0)
    print('time: '+str(time.time()-t_start))
    return sorted_dist


def WordFrequencyDist_D(words):
    """ takes list of words as input
    returns the frequncy of words
    this newer version is much faster than
    the first one"""
    t_start = time.time()
    dist = {words[0]: 1}
    for w in words[1:]:
        if w in dist:
            dist[w] += 1
        else:
            dist[w] = 1 
    print('time: '+str(time.time()-t_start))
    sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    print('time: '+str(time.time()-t_start))
    return sorted_dist
