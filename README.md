
# The Story Clozed Task

Date: 30.05.2018

The target of this NLU project was to successfully accomplish the story
cloze task. This task was developed to measure advanced commonsense understanding within every-
day written stories [1]. To solve the task, one has to determine the correct ending sentence out of two
available options, given the four initial context sentences of the story. 

Two training sets were provided; one consisted of the context sentences with the correct ending only (train: train_stories.csv)
and the second included also an additional incorrect ending (validation 1: cloze_test_spring2016-
test.csv). Further, an evaluation set (validation 2: cloze_test_val__spring2016_cloze_test_ALL_val.csv) with
context sentences and labeled endings was provided to determine the accuracy of the classifier.
Finally, a test set (test: test_nlu18_utf-8.csv) was provided to participate in the competition by classifying
unlabeled sentences.
In the LSDSem 2017 competition, in which different teams competed to solve the story cloze task,
validation accuracies between 0.595 and 0.752 were reached [2]. The major difficulty of this task
is to extract semantic information from the story context and use it to determine the correct ending.


[1] Nasrin Mostafazadeh et al. A corpus and cloze evaluation for deeper understanding commonsense
stories. 2016.

[2] Nasrin Mostafazadeh et al. Lsdsem 2017 shared task: The story cloze test. 2017.

[3] [Story Cloze Test and ROCStories Corpora Description](http://cs.rochester.edu/nlp/rocstories/)


## Getting Started
1. Install dependencies by running: `python3 setup.py install`

2. Run `word2vec.py` to create the word emebeddings. They will be saved and reloaded for further task. The code is taken from: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

3. Run `main_story_clozed_task.py` with the training_mode rnn option set to true. The first run will take a couple minutes because multiple embeddings will have to be loaded and calculated.

4. Run 'main_story_clozed_task.py' with the training_mode rnn option set to false. The model will predict on the validation set and display the validation accuracy. A output file containing predictions on the test set is saved in the current folder.


