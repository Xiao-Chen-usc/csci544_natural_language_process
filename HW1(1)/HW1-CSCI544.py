#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
 


# In[2]:


#! pip install bs4 # in case you don't have it installed

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz


# ## Read Data

# In[3]:


import pandas as pd
test = pd.read_csv("amazon_reviews_us_Kitchen_v1_00.tsv", sep = '\t',error_bad_lines = False)
test['label'] = -1


# ## Keep Reviews and Ratings

# In[4]:



test.label[test.star_rating>3] = 1


# # Labelling Reviews:
# ## The reviews with rating 4,5 are labelled to be 1 and 1,2 are labelled as 0. Discard the reviews with rating 3'

# In[5]:


test.label[test.star_rating>3] = 1
test.label[test.star_rating<3] = 0
test = test[['label','review_body']] 


# In[6]:


show = list(test.groupby('label').count().review_body)


# In[7]:


print('The numbers for 3 classes are:' +str(show[0])+','+str(show[1])+','+str(show[2]))
new_test = test.loc[test['label']!=-1]


#  ## We select 200000 reviews randomly with 100,000 positive and 100,000 negative reviews.
# 
# 

# In[8]:


postive = new_test.loc[test['label']==1].sample(100000)
negative = new_test.loc[test['label']== 0].sample(100000)
new_p_n = pd.concat([postive,negative])


# In[9]:


length_before_cleaning = new_p_n['review_body'].apply(lambda x:len(str(x))).mean()


# # Data Cleaning
# 
# ## Convert the all reviews into the lower case.

# In[10]:


new_p_n['review_body'] = new_p_n['review_body'].str.lower()


# ## remove the HTML and URLs from the reviews

# In[11]:


def tag(x):
    return re.sub('<.*?>','',str(x))
new_p_n['review_body'] = new_p_n['review_body'].apply(lambda x:tag(x))

def url(x):
    return re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]','',str(x))

new_p_n['review_body'] = new_p_n['review_body'].apply(lambda x:url(x))


# ## perform contractions on the reviews.

# In[12]:


import contractions
new_p_n['review_body'] = new_p_n['review_body'].apply(lambda x:contractions.fix(x))


# ## remove non-alphabetical characters

# In[13]:


def non_alphabetical(x):
    return re.sub('[^a-zA-Z\s]','',str(x))

new_p_n['review_body'] = new_p_n['review_body'].apply(lambda x:non_alphabetical(x))


# ## Remove the extra spaces between the words

# In[14]:


def extra_space(x):
    return re.sub( ' +',' ',str(x))
new_p_n['review_body'] = new_p_n['review_body'].apply(lambda x:extra_space(x))


# Average length of reviews before and after data cleaning (with comma between them)

# In[15]:


length_after_cleaning = new_p_n['review_body'].apply(lambda x:len(str(x))).mean()


# In[16]:


print("Average length of reviews before and after data cleaning :"+str(length_before_cleaning)+','+str(length_after_cleaning))


# # Pre-processing

# ## remove the stop words 

# In[17]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words_set = set(stopwords.words('english'))
from nltk import word_tokenize, pos_tag

def stop_words(x):
    word_tokens = word_tokenize(x)
    temp = []
    for i in word_tokens:
        if i not in stop_words_set:
            temp.append(i)
    return temp
new_p_n['review_body'] = new_p_n['review_body'].apply(lambda x:stop_words(x))


# ## perform lemmatization  

# In[18]:


from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatization(x:list):
    tagged_sent = pos_tag(x)
    lemmas_sent = []
    for tag in tagged_sent:
        wnl = WordNetLemmatizer()
        pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos))
    return lemmas_sent

new_p_n['review_body'] = new_p_n['review_body'].apply(lambda x:lemmatization(x))


# In[19]:


length_after_process = new_p_n['review_body'].apply(lambda x:len(str(x))).mean()


# In[20]:


print("Average length of reviews before and after data cleaning :"+str(length_after_cleaning)+','+str(length_after_process))


# # TF-IDF Feature Extraction

# In[21]:


new_p_n['review_str'] = new_p_n['review_body'].apply(lambda x:' '.join(x))
from sklearn.feature_extraction.text import TfidfVectorizer
X = new_p_n['review_str']
Y = new_p_n['label']
v = TfidfVectorizer()
x_tfidf = v.fit_transform(X)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_tfidf, Y, random_state = 19, test_size = 0.2)


# # Perceptron

# In[22]:


from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

y_train_predict = perceptron.predict(x_train)

y_test_predict = perceptron.predict(x_test)


# In[23]:


report_train = classification_report(y_train,y_train_predict,output_dict=True)
precision_train =  report_train['macro avg']['precision'] 
recall_train = report_train['macro avg']['recall']    
f1_train = report_train['macro avg']['f1-score']
accuracy_train = report_train['accuracy']


# In[24]:


report_test = classification_report(y_test,y_test_predict,output_dict=True)
precision_test =  report_train['macro avg']['precision'] 
recall_test = report_train['macro avg']['recall']    
f1_test = report_train['macro avg']['f1-score']
accuracy_test = report_train['accuracy']


# In[38]:


print('Accuracy, Precision, Recall, and f1-score for training and testing split (in the mentioned order)for Perceptron (with comma between them):\n'+       str(accuracy_train)+','+str(precision_train)+','+str(recall_train)+','+      str(f1_train)+','+str(accuracy_test)+','+str(precision_test)+','+str(recall_test)+','+str(f1_train))


# # SVM

# In[39]:


from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(x_train, y_train)
y_train_predict = svc.predict(x_train)
y_test_predict = svc.predict(x_test)


# In[40]:


report_train = classification_report(y_train,y_train_predict,output_dict=True)
precision_train =  report_train['macro avg']['precision'] 
recall_train = report_train['macro avg']['recall']    
f1_train = report_train['macro avg']['f1-score']
accuracy_train = report_train['accuracy']


# In[41]:


report_test = classification_report(y_test,y_test_predict,output_dict=True)
precision_test =  report_train['macro avg']['precision'] 
recall_test = report_train['macro avg']['recall']    
f1_test = report_train['macro avg']['f1-score']
accuracy_test = report_train['accuracy']


# In[42]:


print('Accuracy, Precision, Recall, and f1-score for training and testing split (in the mentioned order) for SVM (with comma between them):\n'+str(accuracy_train)+','+str(precision_train)+','+str(recall_train)+','+str(f1_train)+','+str(accuracy_test)+','+str(precision_test)+','+str(recall_test)+','+str(f1_test))


# # Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state=0,max_iter = 300).fit(x_train, y_train)
y_train_predict = logistic.predict(x_train)
y_test_predict = logistic.predict(x_test)


# In[44]:


report_train = classification_report(y_train,y_train_predict,output_dict=True)
precision_train =  report_train['macro avg']['precision'] 
recall_train = report_train['macro avg']['recall']    
f1_train = report_train['macro avg']['f1-score']
accuracy_train = report_train['accuracy']


# In[45]:


report_test = classification_report(y_test,y_test_predict,output_dict=True)
precision_test =  report_train['macro avg']['precision'] 
recall_test = report_train['macro avg']['recall']    
f1_test = report_train['macro avg']['f1-score']
accuracy_test = report_train['accuracy']


# In[46]:


print('Accuracy, Precision, Recall, and f1-score for training and testing split (in the mentioned order) for Logistic Regression (with comma between them):\n'+str(accuracy_train)+','+str(precision_train)+','+str(recall_train)+','+str(f1_train)+','+str(accuracy_test)+','+str(precision_test)+','+str(recall_test)+','+str(f1_test))


# # Naive Bayes

# In[47]:


from sklearn.naive_bayes import MultinomialNB
NB= MultinomialNB()
NB.fit(x_train, y_train)
y_train_predict = NB.predict(x_train)
y_test_predict = NB.predict(x_test)


# In[48]:


report_train = classification_report(y_train,y_train_predict,output_dict=True)
precision_train =  report_train['macro avg']['precision'] 
recall_train = report_train['macro avg']['recall']    
f1_train = report_train['macro avg']['f1-score']
accuracy_train = report_train['accuracy']


# In[49]:


report_test = classification_report(y_test,y_test_predict,output_dict=True)
precision_test =  report_train['macro avg']['precision'] 
recall_test = report_train['macro avg']['recall']    
f1_test = report_train['macro avg']['f1-score']
accuracy_test = report_train['accuracy']


# In[50]:


print('Accuracy, Precision, Recall, and f1-score for training and testing split (in the mentioned order) for Naive Bayes (with comma between them):\n'+str(accuracy_train)+','+str(precision_train)+','+str(recall_train)+','+str(f1_train)+','+str(accuracy_test)+','+str(precision_test)+','+str(recall_test)+','+str(f1_test))


# In[ ]:




