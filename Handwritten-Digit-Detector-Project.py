# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:49:43 2018

@author: Tamer
"""


from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Analyze the images in dataset to check it

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

print('Training data shape: ', x_train.shape, y_train.shape)
print('Testing data shape: ', x_test.shape, y_test.shape)

# Find the unique numbers from the training label

classes = np.unique(y_train)
n_classes = len(classes)


print('Output classes', classes)
print('Number of outputs "classes" ', n_classes)

# Plot the images in dataset

plt.figure(figsize = [10, 10])
plt.subplot(1, 2, 1)
plt.imshow(x_train[1, :, :], cmap ='gray')
plt.title('Ground Truth: {}'.format(y_train[1]))

plt.subplot(1, 2, 2)
plt.imshow(x_test[1, :, :], cmap = 'gray')
plt.title('Ground Truth: {}'.format(y_test[1]))

# Reshape the image matrix to 1 color 'gray'
x_train = x_train.reshape(-1, 28, 28, 1)

x_test = x_test.reshape(-1, 28, 28, 1)

# Scale the image matrix between 0 and 1
x_train.shape, x_test.shape

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train = x_train / 255

x_test = x_test / 255


# Changing the labels from categorical to one hot encoding

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

# Import packages
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU




classifier = Sequential()

# 32 Number of filters it should increase within the depth of the model
classifier.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1) ))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# After last conv layer and maxpooling and before first fully connected layer add flatten
classifier.add(Flatten())

classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(10, activation = 'sigmoid'))

# After the model is created we will need the gradient descent to compute the cost "J"
# by using optimization algorithm as adam which is stochastic gradient descen algorithm 
# or rmsprop or other so to do this use compile function as below
classifier.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])

classifier.summary()

classifier_train = classifier.fit(x_train, y_train, validation_split = 0.2, batch_size = 64, epochs = 20)


# Evaluate the model first method:

test_eval = classifier.evaluate(x_test, y_test, verbose = 0)

loss = test_eval[0]        # Test Loss:  0.05648441900585672
accuracy = test_eval[1]    # Test accuracy:  0.9909

print('Test Loss: ', loss)
print('Test accuracy: ', accuracy)


# Evaluate the model second method:

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_function():
    classifier = Sequential()
    
    classifier.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1) ))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    classifier.add(Flatten())
    
    classifier.add(Dense(128, activation = 'relu'))
    classifier.add(Dense(10, activation = 'sigmoid'))
    classifier.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_function, batch_size = 64, epochs = 20)
accuracy = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv= 10, n_jobs = 1)

# Plot accuracy and loss between training dnd validation data

history_dict = classifier_train.history
history_dict.keys()

accuracy = classifier_train.history['acc']
val_accuracy = classifier_train.history['val_acc']

loss = classifier_train.history['loss']
val_loss = classifier_train.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label = 'Validation Accuracy')
plt.title('Training and validation accuracy')

plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and validation loss')

plt.legend()
plt.show()


################################################################################

##Create the model after adding dropout##


classifier_dropout = Sequential()

classifier_dropout.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1) ))
classifier_dropout.add(MaxPooling2D(pool_size = (2, 2)))
classifier_dropout.add(Dropout(0.2))

classifier_dropout.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier_dropout.add(Dropout(0.2))

classifier_dropout.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
classifier_dropout.add(MaxPooling2D(pool_size = (2,2)))
classifier_dropout.add(Dropout(0.2))

classifier_dropout.add(Flatten())

classifier_dropout.add(Dense(128, activation = 'relu'))
classifier_dropout.add(Dropout(0.2))

classifier_dropout.add(Dense(10, activation = 'sigmoid'))

classifier_dropout.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])

classifier_dropout.summary()

classifier_train_dropout = classifier_dropout.fit(x_train, y_train, validation_split = 0.2, batch_size = 64, epochs = 20)

# Saving the model

classifier_dropout.save('handwrittent-digit-model.h5py')

accuracy = classifier_train_dropout.history['acc']
val_accuracy = classifier_train_dropout.history['val_acc']

loss = classifier_train_dropout.history['loss']
val_loss = classifier_train_dropout.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label = 'Validation Accuracy')
plt.title('Training and validation accuracy')

plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and validation loss')

plt.legend()
plt.show()