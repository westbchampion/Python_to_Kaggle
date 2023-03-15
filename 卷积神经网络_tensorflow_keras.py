from keras import models,layers,losses,optimizers

HIDDEN_SIZES=256
NUM_CLASSES=10
EPOCHS=5
BATCH_SIZZE=64
LEARNING_RATE=1e-3

#初始化卷积神经网络模型
model=models.Sequential()

model.add(layers.Conv2D(32,(3,3)activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPool2D(2,2))

#初始化