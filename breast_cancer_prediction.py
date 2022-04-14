# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:53:13 2022

@author: xiangkiwi (Github)
"""

import numpy as np
import pandas as pd
import sklearn.datasets as skdatasets

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Read the csv file
data = pd.read_csv(r"C:\Users\USER\Desktop\Python\Deep Learning\Datasets\breast_cancer_data.csv")
#Read the first five rows of data
print(data.head())
print(data.info())

#%%
#Drop the unnecessary columns
data = data.drop('id', axis = 1)
data = data.drop('Unnamed: 32', axis = 1)
#Double check the data after drop the columns
print(data.head())
print(data.info())

#%%
#Mapping the diagnosis category
mapping = {'M' : 0, 'B' : 1}
data['diagnosis'] = data['diagnosis'].map(mapping)
features = list(data.columns[1:31])

#%%
data_features = data[features]
data_labels= data['diagnosis'].values

SEED = 12345

x_train, x_test, y_train, y_test = train_test_split(data_features, data_labels, test_size = 0.2, random_state = SEED)

#Data preparation is done
#%%
import tensorflow as tf
model = tf.keras.Sequential()

nClass = len(np.unique(y_test))

inputs = tf.keras.Input(shape = (x_train.shape[-1],))
dense = tf.keras.layers.Dense(128, activation = 'relu')
x = dense(inputs)
dense = tf.keras.layers.Dense(64, activation = 'relu')
x = dense(x)
dense = tf.keras.layers.Dense(32, activation = 'relu')
x = dense(x)
outputs = tf.keras.layers.Dense(nClass, activation = 'sigmoid')(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'cancer_model')
model.summary()

#%%
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 32, epochs = 20)

#Accuracy is up to 90%
#%%
#Show Loss and Accuracy Graph
import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = history.epoch

plt.plot(epochs, training_loss, label = 'Training Loss')
plt.plot(epochs, val_loss, label = 'Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.figure()

plt.plot(epochs, training_accuracy, label = 'Training accuracy')
plt.plot(epochs, val_accuracy, label = 'Validation accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.figure()

