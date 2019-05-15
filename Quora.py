# -*- coding: utf-8 -*-
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
        
Algorithm for Baseline Model(Logistic Regression):
    Step 1 : Find the target variable to be classified and select the desired regression model such as Binary, Ordinary etc.
    Step 2 : Select the feature and split the data accordingly.We have used k fold cross validation with 5 splits and shuffle is true and random state as 43. 
    Step 3 : We have used features on train data that includes Number of Words in the text, Number of Unique Words in the text, Number of characters in the text,Number of stopwords in the text, Number of punctuations in the text, Number of upper case words in the text, Number of title words in the text and Average length of words in the text
    Step 4 : In the classifier - Logistic Regression we have used class weight = "balance", C=0.5 and solver = 'sag'.
    Step 5 : Perfrom the Model Evaluation using Confusion Matrix and F1 score.
          
Algorithm for Logistic Regression with TF-IDF:
# Step 1 : Find the target variable to be classified and select the desired regression model such as Binary, Ordinary etc.
# Step 2 : Select the feature and split the data accordingly.We have used k fold cross validation with 5 splits and shuffle is true and random state as 50. 
# Step 3 : We have used Tokenization and Padding of sentences, N-grams feature addition, Stopwords removal and Word Vectorization feature addition (TF-IDF Term frequency-inverse document frequency).In addition we have used stripping accents (à, è, ù),stripping punctuations, converting to lowercase, tokenizing, analyzer (word / char), max features and sublinear tf (scaling)
# Step 4 : In the classifier - Logistic Regression we have used class weight = "balance", C=0.5 and solver = 'sag'.
# Step 5 : Perfrom the Model Evaluation using Confusion Matrix and F1 score.
    
Algorithm for Naive Bayes:
    Step 1 : Find the target variable to be classified.
    Step 2 : The result of calling fit on a GridSearchCV object is a classifier that we can use to predict
    Step 3 : Summarize the data and make predictions. 
    Step 4 : Evaluate the model to make predictions and calculate the desire F1 score.
    
Algorithm for NBSVM:
    Step 1 : Split the model into test and train and build a class to fit and verify the model
    Step 2 : Use N-grams and Vectorization which helps us in Converting a Document-Term Matrix to Word ID Sequences
    Step 3 : Use the Optimal threshold and find F1 score.
            
        
"""


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

##import os
##print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import plotly.offline as py
import plotly.graph_objs as go

from collections import defaultdict
from plotly import tools

import string
import seaborn as sns

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

##os.path

#Importing train.csv
quora_train = pd.read_csv('C:/Users/shrey/Desktop/George Mason University/Sem 2/AIT 690/Completed Assignments/Team Project/train.csv')
#Head of train.csv(First 6 rows)
quora_train.head()

#Importing test.csv
quora_test =  pd.read_csv('C:/Users/shrey/Desktop/George Mason University/Sem 2/AIT 690/Completed Assignments/Team Project/test.csv')
#Head of test.csv
quora_test.head()

#Size of Quora Train Dataset
print("Quora Train dataset shape: ", quora_train.shape)
print("Quora Test dataset shape: ", quora_test.shape)

#Quora Train dataset --> class, range index, data columns, data type and memory used
quora_train.info()

#A look at the Target variable
train_target = quora_train['target'].values

#To check if there are any missing values in Quora Train Dataset
quora_train.isnull().any()

#To count Sincere and Insincere questions in Quora Train Dataset
quora_train['target'].value_counts()

#Plot --> count Sincere and Insincere questions in Quora Train Dataset
quora_train.target.value_counts().plot(kind='bar', color='Blue', title='Frequency of Sincere vs Insincere Questions', x='0-Sincere or 1-Insincere Question', y='Number of Questions')

#Pie Distribution Plot using Plotly
labels = (np.array((quora_train['target'].value_counts()).index))
values = (np.array(((quora_train['target'].value_counts()) / (quora_train['target'].value_counts()).sum())*100))

trace = go.Pie(labels=labels, values=values)
layout = go.Layout(
    title='Target distribution',
    font=dict(size=18),
    width=600,
    height=600,
)
data=[trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="usertype.html")

#For simplicity
sincere_questions = quora_train[quora_train['target'] == 0]
insincere_questions = quora_train[quora_train['target'] == 1]

#Pip install wordcloud in Anaconda Prompt
#Word Cloud - To check frequently occuring words in the data
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
words = ' '.join(quora_train['question_text'])
wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,
                      background_color='white',min_font_size=6,
                      width=3000,collocations=False,
                      height=2500
                     ).generate(words)
plt.figure(1,figsize=(16, 16))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Word Cloud of Train Questions")
plt.show()

# Sincere Word Cloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
words = ' '.join(sincere_questions['question_text'])
wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,
                      background_color='white',min_font_size=6,
                      width=3000,collocations=False,
                      height=2500
                     ).generate(words)
plt.figure(1,figsize=(16, 16))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Word Cloud of Sincere Train Questions")
plt.show()

# Insincere Word Cloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
words = ' '.join(insincere_questions['question_text'])
wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,
                      background_color='white',min_font_size=6,
                      width=3000,collocations=False,
                      height=2500
                     ).generate(words)
plt.figure(1,figsize=(16, 16))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Word Cloud of Insincere Train Questions")
plt.show()

## function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## Get the top 10 frequent words from sincere questions when doing Unigram##
freq_dict = defaultdict(int)
for sent in sincere_questions["question_text"]:
    for word in generate_ngrams(sent,1):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
fd_sorted.head(10)

## Get the top 10 frequent words from insincere questions when doing Unigram##
freq_dict = defaultdict(int)
for sent in insincere_questions["question_text"]:
    for word in generate_ngrams(sent,1):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
fd_sorted.head(10)

## Get the top 10 frequent words from sincere questions when doing Bigram##
freq_dict = defaultdict(int)
for sent in sincere_questions["question_text"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
fd_sorted.head(10)

## Get the top 10 frequent words from insincere questions when doing Bigram##
freq_dict = defaultdict(int)
for sent in insincere_questions["question_text"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
fd_sorted.head(10)

## Get the top 10 frequent words from sincere questions when doing Trigram##
freq_dict = defaultdict(int)
for sent in sincere_questions["question_text"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
fd_sorted.head(10)

## Get the top 10 frequent words from sincere questions when doing Trigram##
freq_dict = defaultdict(int)
for sent in insincere_questions["question_text"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
fd_sorted.head(10)

##Features
## Number of words in the text ##
quora_train["num_words"] = quora_train["question_text"].apply(lambda x: len(str(x).split()))
quora_test["num_words"] = quora_test["question_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
quora_train["num_unique_words"] = quora_train["question_text"].apply(lambda x: len(set(str(x).split())))
quora_test["num_unique_words"] = quora_test["question_text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
quora_train["num_chars"] = quora_train["question_text"].apply(lambda x: len(str(x)))
quora_test["num_chars"] = quora_test["question_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
quora_train["num_stopwords"] = quora_train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
quora_test["num_stopwords"] = quora_test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

## Number of punctuations in the text ##
quora_train["num_punctuations"] =quora_train['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
quora_test["num_punctuations"] =quora_test['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
quora_train["num_words_title"] = quora_train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
quora_test["num_words_title"] = quora_test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Number of title case words in the text ##
quora_train["num_words_upper"] = quora_train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
quora_test["num_words_upper"] = quora_test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Average length of the words in the text ##
quora_train["mean_word_len"] = quora_train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
quora_test["mean_word_len"] = quora_test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

freq, axes = plt.subplots(4, 1, figsize=(10,25))
sns.boxplot(x='target', y='num_words', data=quora_train, ax=axes[0])
axes[0].set_xlabel('Target', fontsize=10)
axes[0].set_title("Number of words in each class", fontsize=12)

sns.boxplot(x='target', y='num_unique_words', data=quora_train, ax=axes[1])
axes[1].set_xlabel('Target', fontsize=10)
axes[1].set_title("Number of words in each class", fontsize=12)

sns.boxplot(x='target', y='num_chars', data=quora_train, ax=axes[2])
axes[2].set_xlabel('Target', fontsize=10)
axes[2].set_title("Number of characters in each class", fontsize=12)

sns.boxplot(x='target', y='num_stopwords', data=quora_train, ax=axes[3])
axes[3].set_xlabel('Target', fontsize=10)
axes[3].set_title("Number of characters in each class", fontsize=12)

freq, axes = plt.subplots(4, 1, figsize=(10,25))
sns.boxplot(x='target', y='num_punctuations', data=quora_train, ax=axes[0])
axes[0].set_xlabel('Target', fontsize=10)
axes[0].set_title("Number of punctuations in each class", fontsize=12)

sns.boxplot(x='target', y='num_words_upper', data=quora_train, ax=axes[1])
axes[1].set_xlabel('Target', fontsize=10)
axes[1].set_title("Number of punctuations in each class", fontsize=12)

sns.boxplot(x='target', y='num_words_title', data=quora_train, ax=axes[2])
axes[2].set_xlabel('Target', fontsize=10)
axes[2].set_title("Number of punctuations in each class", fontsize=12)

sns.boxplot(x='target', y='mean_word_len', data=quora_train, ax=axes[3])
axes[3].set_xlabel('Target', fontsize=10)
axes[3].set_title("Number of punctuations in each class", fontsize=12)
plt.show()


eng_features = ['num_words', 'num_unique_words', 'num_chars', 
                'num_stopwords', 'num_punctuations', 'num_words_upper', 
                'num_words_title', 'mean_word_len']

##Simple Logistic Regression --- Baseline Model  
kf = KFold(n_splits=5, shuffle=True, random_state=43)
test_pred = 0
oof_pred = np.zeros([quora_train.shape[0],])

x_test = quora_test[eng_features].values
for i, (train_index, val_index) in tqdm(enumerate(kf.split(quora_train))):
    x_train, x_val = quora_train.loc[train_index][eng_features].values, quora_train.loc[val_index][eng_features].values
    y_train, y_val = train_target[train_index], train_target[val_index]
    classifier = LogisticRegression(class_weight = "balanced", C=0.5, solver='sag')
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    preds = classifier.predict_proba(x_test)[:,1]
    test_pred += 0.2*preds
    oof_pred[val_index] = val_preds
    
pred_train = (oof_pred > 0.68).astype(np.int)
f1 = f1_score(train_target, pred_train)
score = accuracy_score(train_target, pred_train)
print('Accuracy of Basic Logistic Regression model is:',score)
print('F score  of Basic Logistic Regression model is:',f1)

confusion_matrix(train_target, pred_train)

##############################################################################################

##Logistic Regression with TF-IDF vectors (n-grams)
train_text = quora_train['question_text']
test_text = quora_test['question_text']
all_text = pd.concat([train_text, test_text])

#Features
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 4),
    max_features=5000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

#Model Building...
kf = KFold(n_splits=5, shuffle=True, random_state=50)
test_pred_tf = 0
oof_pred_tf = np.zeros([quora_train.shape[0],])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(quora_train.question_text, quora_train.target, test_size=0.2, random_state=100)

for i, (train_index, val_index) in tqdm(enumerate(kf.split(quora_train))):
    x_train, x_val = train_word_features[train_index,:], train_word_features[val_index,:]
    y_train, y_val = train_target[train_index], train_target[val_index]
    classifier = LogisticRegression(class_weight = "balanced", C=0.5, solver='sag')
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    preds = classifier.predict_proba(test_word_features)[:,1]
    test_pred_tf += 0.2*preds
    oof_pred_tf[val_index] = val_preds

##Selecting the best threshold for higher F1 score on validation sample
pred_train = (oof_pred_tf > 0.8).astype(np.int)
f1 = f1_score(train_target, pred_train)
score = accuracy_score(train_target, pred_train)
print('Accuracy of Logistic Regression model with TF-IDF is:',score)
print('F score  of Logistic Regression model with TF-IDF is:',f1)

confusion_matrix(train_target, pred_train)

####################################################################################################

#Naive Bayes Model
from sklearn.model_selection import train_test_split
quora_train, cv= train_test_split(quora_train, test_size=0.2)
x_train=quora_train.drop(['target'],axis=1)
y_train=quora_train['target']
x_cv=cv.drop(['target'],axis=1)
y_cv=cv['target']

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vect = TfidfVectorizer()
reviews_tfidf = tf_idf_vect.fit_transform(x_train['question_text'].values)
reviews_tfidf1 = tf_idf_vect.transform(x_cv['question_text'].values)
reviews_tfidf2 = tf_idf_vect.transform(quora_test['question_text'].values)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB()
param_grid = {'alpha':[1000,100,10,1,0.1,0.01,0.001]} #params we need to try on classifier
gsv = GridSearchCV(nb,param_grid,cv=2,verbose=1,n_jobs=-1,scoring='f1')
gsv.fit(reviews_tfidf,y_train)
nb = BernoulliNB(alpha=0.1)
nb.fit(reviews_tfidf,y_train)
train_pred = nb.predict(reviews_tfidf)
cv_pred = nb.predict(reviews_tfidf1)

test_pred = nb.predict(reviews_tfidf2)
print("Train Set Accuracy: {}".format(accuracy_score(train_pred, y_train)))
print("Train Set ROC: {}".format(roc_auc_score(train_pred, y_train)))
print("Train Set F1 Score: {}\n".format(f1_score(train_pred, y_train)))
print("Validation Set Accuracy: {}".format(accuracy_score(cv_pred, y_cv)))
print("Validation Set ROC: {}".format(roc_auc_score(cv_pred, y_cv)))
print("Validation Set F1 Score: {}\n".format(f1_score(cv_pred, y_cv)))
print("Confusion Matrix of test set:\n [ [TN  FP]\n [FN TP] ]\n")
df_cm = pd.DataFrame(confusion_matrix(y_val, val_preds.round()), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g')

#################################################################################################

#NBSVM

import re
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(quora_train, test_size=0.07, random_state=2018)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from scipy import sparse
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1, solver='sag'):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.solver = solver
        
    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs, solver=self.solver).fit(x_nb, y)
        return self

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

N = 50000

vec = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1, max_features=N)

trn_term_doc = vec.fit_transform(train_df['question_text'])
val_term_doc = vec.transform(val_df['question_text'])
test_term_doc = vec.transform(quora_test['question_text'])

model = NbSvmClassifier(dual=True, solver='liblinear', C = 1e1)
model.fit(trn_term_doc, train_df['target'])

preds_val = model.predict_proba(val_term_doc)[:,1]
preds_test = model.predict_proba(test_term_doc)[:,1]

best_threshold = threshold_search(y_true=val_df['target'], y_proba=preds_val)
best_threshold
