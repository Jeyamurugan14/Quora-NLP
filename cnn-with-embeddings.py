"""
Final Project: Quora Insincere Questions Classification (Competition available on Kaggle)
Team Name: TeamBots
Team Members: Abhishek Shambhu, Jeyamurugan Krishnakumar, Shreyans Singh

About Quora and the dataset:
    An existential problem for any major website today is how to handle toxic and divisive content. 
    Quora wants to tackle this problem head-on to keep their platform a place where users can feel 
    safe sharing their knowledge with the world.

    Quora is a platform that empowers people to learn from each other. 
    On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. 
    A key challenge is to weed out insincere questions -- those founded upon false premises, 
    or that intend to make a statement rather than look for helpful answers.    

Description and Problem Statement:
    An insincere question is defined as a question intended to make a statement rather than look for helpful answers. Some characteristics that can signify that a question is insincere:

        Has a non-neutral tone
            Has an exaggerated tone to underscore a point about a group of people
            Is rhetorical and meant to imply a statement about a group of people
        Is disparaging or inflammatory
            Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype
            Makes disparaging attacks/insults against a specific person or group of people
            Based on an outlandish premise about a group of people
            Disparages against a characteristic that is not fixable and not measurable
        Isn't grounded in reality
            Based on false information, or contains absurd assumptions
        Uses sexual content (incest, bestiality, pedophilia) for shock value, and not to seek genuine answers
    
    The training data includes the question that was asked, and whether it was identified as insincere (target = 1). 
    The ground-truth labels contain some amount of noise: they are not guaranteed to be perfect.
    
    File descriptions
        train.csv - the training set
        test.csv - the test set
        sample_submission.csv - A sample submission in the correct format
        embeddings -
                GoogleNews-vectors-negative300 - https://code.google.com/archive/p/word2vec/
                glove.840B.300d - https://nlp.stanford.edu/projects/glove/
                paragram_300_sl999 - https://cogcomp.org/page/resource_view/106
                wiki-news-300d-1M - https://fasttext.cc/docs/en/english-vectors.html

    Data fields
        qid - unique question identifier
        question_text - Quora question text
        target - a question labeled "insincere" has a value of 1, otherwise 0
  
Performance Metric Used: F1-score

Process Followed: EDA --> NLP --> Building different Models --> Evaluating results based on score from Kaggle.

EDA Performed: 
    1. Word Cloud Visualization for top 500 words for both Sincere and Insincere Questions
    2. Unigram, Bigram and Trigram Visualization for top 10 Sincere and Insincere Questions 
    3. Classification of data based on Pie Chart and Bar plot for Sincere and Insincere Questions

NLP performed:
    Number of Words in the text
    Number of Unique Words in the text
    Number of characters in the text
    Number of stopwords in the text
    Number of punctuations in the text
    Number of upper case words in the text
    Number of title words in the text
    Average length of words in the text
    Tokenization and Padding of sentences
    N-grams feature addition 
    Stopwords removal 
    Word Vectorization feature addition (TF-IDF - Term frequency-inverse document frequency)

Models Used:  
1. Baseline Model - Logistic Regression
2. Logistic Regression with TF-IDF Vectorization
3. Naive Bayes with TF-IDF Vectorization
4. NBSVM Model with TF-IDF Vectorization
5. CNN Model with Glove Embedding

Algorithm:
    Step 1: We use the Glove embedding (tried with other embeddings too but Glove with CNN gave the highest F1 score.)
    Step 2: We did few considerations before applying to the model:
                max_features = 40000, maxlen = 70, embed_size = 30 and considering threshold as 0.35
    Step 3: Tokeizing and padding the sentences and doing word vectorization
    Step 4: Building CNN Model (4 Conv2D layer, 4 MaxPooling Layer, Flatten, Dropout , Dense Output)
    Step 5: Activation Function: Tanh for the Conv2D layers and Sigmoid activation function for the final layer.
            Optimizer used: Adam Optimizer
            No if epoch: 2
            Batch size - 256
    Step 6: We apply this model on the test data to get an F1 score of 0.673. (Top 10% on Kaggle)
    
    
    
"""
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import warnings
warnings.filterwarnings('ignore')

EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')

X_train = train["question_text"].fillna("fillna").values
y_train = train["target"].values
X_test = test["question_text"].fillna("fillna").values

max_features = 40000
maxlen = 70
embed_size = 300

threshold = 0.35

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
class F1Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred = (y_pred > threshold).astype(int)
            score = f1_score(self.y_val, y_pred)
            print("\n F1 Score - epoch: %d - score: %.6f \n" % (epoch+1, score))
            
filter_sizes = [1,2,3,5]
num_filters = 42

def get_model():    
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#    x = SpatialDropout1D(0.4)(x)
    x = Reshape((maxlen, embed_size, 1))(x)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size),
                                 kernel_initializer='he_normal', activation='tanh')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size),
                                 kernel_initializer='he_normal', activation='tanh')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), 
                                 kernel_initializer='he_normal', activation='tanh')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size),
                                 kernel_initializer='he_normal', activation='tanh')(x)
    
    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = Dropout(0.1)(z)
        
    outp = Dense(1, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()

batch_size = 256
epochs = 2

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95,
                                              random_state=233)
F1_Score = F1Evaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs,
                 validation_data=(X_val, y_val),
                 callbacks=[F1_Score], verbose=2)
                 
y_pred = model.predict(x_test, batch_size=1024)
y_pred = (y_pred > threshold).astype(int)
submission['prediction'] = y_pred
submission.to_csv('submission.csv', index=False)
