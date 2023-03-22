
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D
from sklearn.metrics import log_loss
from keras.models import Model
import cv2

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def vgg16_model( num_classes=None):
	
    model = VGG16(weights='imagenet', include_top=True)
    model.layers.pop()
    #获取最后一层的输出
    #修改模型的输出为当前的最后一层，因为前面已经把模型的输出层pop掉了
    output = model.layers[-1].output
    #添加softmax层
    x = Dense(num_classes, activation='softmax')(output)
    model = Model(inputs=model.input, outputs=x)
    #冻结前8层，不训练
    for layer in model.layers[:8]:
        layer.trainable = False

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

#加载数据集
from keras.datasets import mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x[:10000]
train_y = train_y[:10000]
train_x = np.array(train_x, dtype="uint8")

#将灰度图中的单通道转为RGB多通道
train_x = [cv2.cvtColor(cv2.resize(x, (224, 224)), cv2.COLOR_GRAY2BGR) for x in train_x]
train_x = np.concatenate([arr[np.newaxis] for arr in train_x]).astype('float32')

train_y = train_y.reshape(len(train_y),1).astype(int)
train_y = convert_to_one_hot(train_y,10)

print("训练集：")
print(train_x.shape)
print(train_y.shape)

model = vgg16_model(num_classes=10)
model.fit(train_x, train_y, batch_size=64, epochs=20, shuffle=True)
model.save("mnist_model.h5")

