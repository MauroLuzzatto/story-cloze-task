
# README Story Clozed Task, Group 13
# Thomas Brunschwiler, Dario Kneubühler, Mauro Luzzatto

## Setup


1. Install dependencies by running (with `sudo` appended if necessary)
```
python3 setup.py install
```
2. Run `word2vec.py` to create the word emebeddings. They will be saved and reloaded for further task. The code is taken from: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

3. Run `main_story_clozed_task.py` with the training_mode rnn option set to true. The first run will take a couple minutes because multiple embeddings will have to be loaded and calculated.

4. Run 'main_story_clozed_task.py' with the training_mode rnn option set to false. The model will predict on the validation set and display the validation accuracy. A output file containing predictions on the test set is saved in the current folder.


