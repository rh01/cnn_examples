import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
import scipy.io as sio
import matplotlib.pyplot as plt

import time

from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.optimizers import SGD


mat = sio.loadmat('data200.mat')

data = mat['source']
print(data)
print("*"*100)
print("The shape of Source data is (%s,%s)" % data.shape)
print("*"*100)

labels = mat['target']
print(labels)
print("*"*100)
print("The shape of Label is (%s,%s)" % labels.shape)
print("*"*100)




data = np.array(data.transpose())
labels = np.array(labels.transpose())

print(data.shape)
print(labels.shape)


image_array = np.reshape(data, (len(data), 32, 32, 1))
print(labels.shape)
print(image_array.shape)

X_train, X_test, y_train, y_test = train_test_split(image_array, labels, test_size=0.20)


print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)


model = Sequential()



print( 'Training...')

# CONVOLUTION

# When using THEANO backend
# model.add(Reshape((1, 120, 320), input_shape=(120, 320)))

# # When using TENSORFLOW backend
# model.add(Reshape((120, 320, 1), input_shape=(120, 320)))

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(32 ,32, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

##model.add(Convolution2D(64, 3, 3))
##model.add(Activation('relu'))
##model.add(MaxPooling2D(pool_size=(2, 2)))
##
model.add(Flatten())

model.add(Dense(30, init='uniform'))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(62, init='uniform'))
model.add(Activation('softmax'))


model.summary()
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# # train 
# for step in range(301):
#     cost = model.train_on_batch(X_train, y_train)
#     if step %  100 ==0:
#         print('train cost ', cost)
# Fit the model
model.fit(X_train, y_train,
          nb_epoch=50,
          batch_size=100)


score = model.evaluate(X_test, y_test, batch_size=1000)
loss = score[0]
accuracy = score[1]
print( '')
print ('Loss score: ', loss)
print ('Accuracy score: ', accuracy)


model.save('cnn.h5')

json_string = model.to_json()
with open('.nn_1.json','w') as new_json:
        json.dump(json_string, new_json)







