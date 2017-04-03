
# coding: utf-8

# # Fake Spam Detector

# *By Siddharth Shankar Samal : sidhuking07@gmail.com*

# Basically I am importing the datasets and storing the datas in three columns : 
# *Polarity of the review
# *Review itself
# *True or Deceptive as ('t' or 'd')
# 
# Then I am converting 't' to 1 and 'd' to 0 because I will be using this as my target value and the review as my feature.
# Then I am splitting the Review data into testing data and training data (0.3 and 0.7 respectively).
# Then I am using CountVectorizer() to extract numeric features of each of the review as classifier can only use numeric data to compute something.
# Then I am using MultinomialNB method classifier to classify the reviews as Deceptive/True.

# Dependencies I used :
# * os for loading os folder paths
# * pandas for making dataframes
# * numpy for making arrays
# * sklearn.metrics for accuracy score, precision score, recall score, f1 score
# * sklearn.cross_validation for splitting the dataset
# * CountVectorizer() for extracting features from text in numerical form
# * MultinomialNB for importing naive bayes multinomial method classifier

# **Importing all the dependencies that will be needed.**
# 

# In[1]:

import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split


# **Setting up the folder paths in which the dataset is presetn**
# 

# In[2]:

neg_deceptive_folder_path = 'op_spam_v1.4\\negative_polarity\\deceptive_from_MTurk\\'
neg_true_folder_path = 'op_spam_v1.4\\negative_polarity\\truthful_from_Web\\'
pos_deceptive_folder_path = 'op_spam_v1.4\\positive_polarity\\deceptive_from_MTurk\\'
pos_true_folder_path = 'op_spam_v1.4\\positive_polarity\\truthful_from_TripAdvisor\\'


# **Initialising the lists in which the polarity, review and either it's fake or true will be stored**

# In[3]:

polarity_class = []
reviews = []
spamity_class =[]


# ** Since we have 5 folders in each folder in our dataset, I am using a for loop to iterate through each of the folder and collect datas (i.e Polarity, Review, Fake or True) and store**

# In[4]:

for i in range(1,6):
    insideptru = pos_true_folder_path + 'fold' + str(i) 
    insidepdec = pos_deceptive_folder_path + 'fold' + str(i)
    insidentru = neg_true_folder_path + 'fold' + str(i) 
    insidendec = neg_deceptive_folder_path + 'fold' + str(i) 
    pos_list = []
    for data_file in sorted(os.listdir(insidendec)):
        polarity_class.append('negtive')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidendec, data_file)) as f:
                contents = f.read()
                reviews.append(contents)
    for data_file in sorted(os.listdir(insidentru)):
        polarity_class.append('negative')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidentru, data_file)) as f:
                contents = f.read()
                reviews.append(contents)
    for data_file in sorted(os.listdir(insidepdec)):
        polarity_class.append('positive')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidepdec, data_file)) as f:
                contents = f.read()
                reviews.append(contents)
    for data_file in sorted(os.listdir(insideptru)):
        polarity_class.append('positive')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insideptru, data_file)) as f:
                contents = f.read()
                reviews.append(contents)


# ** Making the dataframe using pandas to store polarity, reviews and true or fake **

# *Setting '0' for deceptive review and '1' for true review*

# In[5]:

data_fm = pd.DataFrame({'polarity_class':polarity_class,'review':reviews,'spamity_class':spamity_class})

data_fm.loc[data_fm['spamity_class']=='d','spamity_class']=0
data_fm.loc[data_fm['spamity_class']=='t','spamity_class']=1


# ** Splitting the dataset to training and testing (0.7 and 0.3)**

# In[6]:

data_x = data_fm['review']

data_y = np.asarray(data_fm['spamity_class'],dtype=int)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_x, data_y, test_size=0.3)




# ** Using CountVectorizer() method to extract features from the text reviews and convert it to numeric data, which can be used to train the classifier **

# *Using fit_transform() for X_train and only using transform() for X_test*

# In[7]:

cv =  CountVectorizer()

X_traincv = cv.fit_transform(X_train)
X_testcv = cv.transform(X_test)


# **Using Naive Bayes Multinomial method as the classifier and training the data**

# In[8]:

nbayes = MultinomialNB()

nbayes.fit(X_traincv, y_train)


# **Predicting the fake or deceptive reviews**
# 

# *using X_testcv : which is vectorized such that the dimensions are matched*

# In[9]:

y_predictions = nbayes.predict(X_testcv)


# ** Printing out fake or deceptive reviews **

# In[10]:

y_result = list(y_predictions)
yp=["True" if a==1 else "Deceptive" for a in y_result]
X_testlist = list(X_test)
output_fm = pd.DataFrame({'Review':X_testlist ,'True(1)/Deceptive(0)':yp})
output_fm


# ** Printing out the Accuracy, Precision Score, Recall Score, F1 Score **

# In[11]:

print("Accuracy % :",metrics.accuracy_score(y_test, y_predictions)*100)
print("Precision Score: ", precision_score(y_test, y_predictions, average='micro'))
print("Recall Score: ",recall_score(y_test, y_predictions, average='micro') )
print("F1 Score: ",f1_score(y_test, y_predictions, average='micro') )

