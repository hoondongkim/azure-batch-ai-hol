
# coding: utf-8

# # Bidirectional-LSTM Text Classification Simple Deep Learning Model

# ### Import Module

# In[1]:


from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

### for Horovod
from keras import optimizers
import math
import horovod.keras as hvd

### Initialize Horovod.
hvd.init()

# ### member variable

# In[2]:


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32
epoch_num = 4
## Horovod epoch_num 재정의
epoch_num = int(math.ceil(epoch_num / hvd.size()))
# Adjust learning rate based on number of GPUs.
# opt = optimizers.Adadelta(1.0 * hvd.size())
opt = optimizers.Adadelta(1.0 * hvd.size())

# Add Horovod Distributed Optimizer.
#opt = hvd.DistributedOptimizer(opt)
opt = hvd.DistributedOptimizer(opt, device_dense='/cpu:0')

# ### Load Data

# In[3]:


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


# ### Text Data Pre-processing

# In[4]:


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)


# ### LSTM Model Define

# In[5]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# ### Model Compile

# In[6]:


model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# ### Model Train

# In[7]:
# Horovod Call Back
callBackList = [
        #EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'),
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch_num,
          validation_data=(x_test, y_test),
          callbacks=callBackList)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)


# ### Accuracy Check 

# In[8]:


print('Test score:', score)
print('Test accuracy:', acc)

