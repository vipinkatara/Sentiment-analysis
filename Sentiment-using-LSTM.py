#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords


# # Loading the data

# In[2]:


df = pd.read_csv('training.1600000.processed.noemoticon.csv', engine='python', names=['target', 'ids', 'date', 'flag', 'user', 'text'])
df.head()


# In[3]:


df['target'].unique()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.shape


# In[7]:


df.drop_duplicates(inplace=True)


# In[8]:


df.shape


# In[9]:


df['flag'].unique()


# In[10]:


df.drop('flag', axis=1, inplace=True)


# In[11]:


print(df['ids'].value_counts())


# In[12]:


print(df)


# In[13]:


df['date'] = pd.to_datetime(df['date'],origin='unix')


# In[14]:


print(df)


# In[15]:


sns.countplot(df['target'])
plt.show()


# In[16]:


df['target'] = df['target'].map({0:0, 4:1})


# In[17]:


for i in df['text'].values:
    if(len(re.findall('.<*?>', i))):
        print(i)
        print('\n')


# In[18]:


stopwords = set(stopwords.words('english'))
stopwords


# In[19]:


stopwords.difference_update({'against', 'ain','aren',
 "aren't",'couldn',
 "couldn't",'couldn',
 "couldn't",'didn',
 "didn't",'don',
 "don't",'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",'isn',
 "isn't",'mightn',
 "mightn't",'mustn',
 "mustn't",'needn',
 "needn't",
 'no',
 'nor',
 'not','shouldn',
 "shouldn't",'weren',
 "weren't",'won',
 "won't",
 'wouldn',
 "wouldn't","doesn't"})


# In[20]:


stopwords


# In[21]:


def cleanpunc(sentences):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentences)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned


# In[22]:


from nltk.stem import SnowballStemmer
sno = SnowballStemmer('english')


# In[23]:


str1=' '
s=' '
i = 0
final_string = []
for wor in df['text'].values:
    fil_wor = []
    for w in wor.split():
        for cleanedwords in cleanpunc(w).split(): #cleaning punctuation
            if(cleanedwords.isalpha() and len(cleanedwords)>2): #checking value is alpha numeric or not and we know adjective size is greater than 2
                if(cleanedwords.lower() not in stopwords):
                    s=(sno.stem(cleanedwords.lower())).encode('utf8') #applying stemmer and converting the character to lowercase
                    fil_wor.append(s)
    str1 = b" ".join(fil_wor) #final string of cleaned words
    #print("***********************************************************************")
    
    final_string.append(str1)
    i+=1


# In[24]:


#copying the column to exixsting dataset
df['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review


# In[25]:


from wordcloud import WordCloud
spam_words = ' '.join(list(df[df['target'] == 0]['text']))
spam_wc = WordCloud(width = 512, height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.show()


# In[27]:


spam_words = ' '.join(list(df[df['target'] == 1]['text']))
spam_wc = WordCloud(width = 512, height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.show()


# In[28]:


df


# In[29]:


Y = df['target']
df['CleanedText'] = df['CleanedText'].apply(str)


# In[30]:


df['target'].map({0 : 0, 4: 1})


# In[31]:


from keras.utils import to_categorical
Y = to_categorical(Y)


# In[32]:


from keras.preprocessing.text import Tokenizer
tkn = Tokenizer(nb_words=2000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')

tkn.fit_on_texts(df['CleanedText'].values)
from keras.preprocessing.sequence import pad_sequences


X = tkn.texts_to_sequences(df['CleanedText'].values)
X = pad_sequences(X)


# In[33]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[34]:


from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
embed_dim = 128
lstm_out = 196
max_fatures = 2000

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[35]:


batch_size=2
model.fit(x_train, y_train,validation_split=0.2, epochs = 30, batch_size=batch_size)


# In[ ]:


from sklearn.metrics import accuracy_score
pred = model.predict(x_test)
print('Accuracy : ',accuracy_score(y_test, pred.round()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




