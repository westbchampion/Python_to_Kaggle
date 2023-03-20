import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
#from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras import models,layers,losses,optimizers
#图像数据生成
from keras.preprocessing.image import ImageDataGenerator
#载入预训练模型：
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
import numpy as np
import os,warnings
#print(tensorflow.__version__)
#print(tensorflow.test.is_gpu_available())
warnings.filterwarnings('ignore')

(X_train,y_train),(X_test,y_test)=cifar10.load_data()

print(X_train.shape)#检查输入数据形状
print(y_train.shape)

#数据处理准则：图像数据做归一化；数值数据做标准化；文本数据做Embeding独热编码

classes=["airplane","automobile","bird","cat","dear","dog","frog","horse","ship","trunk"]
HIDDEN_SIZES=3000
HIDDEN_SIZES=64
NUM_CLASSES=10
EPOCHS=5
BATCH_SIZZE=64
LEARNING_RATE=1e-3


def show_image(X,index):
    plt.figure(figsize=(10,8))
    plt.imshow(X[index])

#图像处理：
"""
datagen=ImageDataGenerator(
    
    
)
generator=datagen.flow_from_directory(
    
)

"""
#载入预训练模型:VGG16，不含全连接层
#model=VGG16(weights='imagenet',include_top=False)

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(HIDDEN_SIZES,activation='relu'))
model.add(layers.Dense(NUM_CLASSES,activation='softmax'))

model.compile(
    optimizer=optimizers.Adam(LEARNING_RATE),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

"""
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.sparse_categorical_accuracy])
model.compile(optimizer='rmsprop',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
optimizers:
SGD() (with or without momentum)
RMSprop()
Adam()
etc.

loss:
MeanSquaredError()
KLDivergence()
CosineSimilarity()
etc.

metrics:
AUC()
Precision()
Recall()
etc.

"""

model.summary()
history=model.fit(X_train,y_train,epochs=25,validation_split=0.2)

#model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1, steps_per_epoch=1)

#model.save('CIFAR10.h5')

result=model.evaluate(X_test,y_test,verbose=0)
print('卷积神经网络在cifar10数据集上的准确率为%.2f%%'%(result[1]*100))
print('卷积神经网络在cifar10数据集上的loss为%.2f'%(result[0]))

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()

#预测
#y_prediction=model.predict(X_test)
 
pred = model.predict(X_test) 
pred = np.argmax(pred, axis = 1)[:10] 
label = np.argmax(y_test,axis = 1)[:10] 
 
print(pred) 
print(label)
#pred=label则能成功预测前10个图像


#show_image(y_prediction,0)

#建立评估指标：
#tensorboard查看训练效果 tensorboard --logdir=log
