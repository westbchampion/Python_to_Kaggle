from keras import layers,models,optimizers,losses
from keras.datasets import cifar10

import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test)=cifar10.load_data()

print(X_train.shape)
print(y_train.shape)
HIDDEN_SIZE=256#要有下划线
NUM_CLASSES=10#要有下划线避免语法重叠
LEARNING_RATE=1E-3
model=models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(HIDDEN_SIZE,activation='relu'))
model.add(layers.Dense(NUM_CLASSES,activation='softmax'))

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

EPOCH=10
history=model.fit(X_train,y_train,epochs=EPOCH,validation_split=0.2)
#pd.DataFrame(history.history).plot(figsize=(8,5))
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()

result=model.evaluate(X_train,y_train,verbose=0)
print('卷积神经网络在cifar10数据集上的准确率为%.2f%%'%(result[1]*100))
print('卷积神经网络在cifar10数据集上的loss为%.2f'%(result[0]))

import numpy as np
pred=model.predict(X_test)
pred = np.argmax(pred, axis = 1)[:10] 
label = np.argmax(y_test,axis = 1)[:10] 
 
print(pred)
print(label)

model.save('model2.h5')
