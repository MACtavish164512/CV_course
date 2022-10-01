import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D

#定义参数
testRatio = 0.2 #学习率
valRatio = 0.2
imageDimensions= (32,32,3)
batchSizeVal= 5
epochsVal = 15
stepsPerEpochVal = 1200

#加载数据集
images = [] #存放所有的图片
classNO = [] #存放图片表示的类别
mylist = os.listdir("myData")
#print(mylist)
noOFclass = len(mylist)
print("类别总数为：",noOFclass)
print("加载数据集")

#遍历数据并存放在列表当中
for x in range(0,noOFclass):
  myPiclist = os.listdir("myData" + "/" + str(x))
  for y in myPiclist:
     curimg = cv2.imread("myData" + "/" + str(x) + "/" + y)
     curimg = cv2.resize(curimg,(32,32))
     images.append(curimg)
     classNO.append(x)
  print(x,end=" ")
print("读取的图片总数为:",len(images))

#将图片转换为numpy的形式
images = np.array(images)
classNO = np.array(classNO)
#print(images.shape)

#数据分割 分成训练集 测试集 验证集
X_train,X_test,y_train,y_test = train_test_split(images,classNO,test_size=0.2)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.2)

print("训练集个数为:",len(X_train))
print("验证集个数为:",len(X_validation))
print("测试集个数为:",len(X_test))
#print(np.where(y_train == 0))

#获取每一个类别的数量
numofclass = []
for x in range(0,noOFclass):
    numofclass.append(len(np.where(y_train == x)[0]))

plt.figure(figsize=(10,5))
plt.bar(range(0,noOFclass,1),numofclass)
plt.title("no of each class")
plt.xlabel("class no")
plt.ylabel('Number of Images')
plt.show()

#图片的预处理 把图片转换为灰度图 and 图像均衡化
def preProcess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(preProcess,X_train)))
X_test = np.array(list(map(preProcess,X_test)))
X_validation = np.array(list(map(preProcess,X_validation)))
#print(X_train[0].shape)

#查看图片
# img = X_train[0]
# img = cv2.resize(img,(300,300))
# cv2.imshow("output",img)
# cv2.waitKey(0)

#图像重塑
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
#print(X_train.shape)

#图像数据的增强
dataGen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,
                             shear_range=0.1,rotation_range=10)
dataGen.fit(X_train)

#独热编码
y_train = to_categorical(y_train,noOFclass)
y_test = to_categorical(y_test,noOFclass)
y_validation = to_categorical(y_validation,noOFclass)

#设计一个训练模型
#设计训练模型
def myModel():
    noOfFilters = 60 #过滤器
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                               imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool)) #保留主要特征的同时进行降维
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5)) #减少过拟合

    model.add(Flatten()) #用来将输入“压平”，即把多维的输入一维化
    model.add(Dense(noOfNodes, activation='relu')) #全连接层
    model.add(Dropout(0.5))
    model.add(Dense(noOFclass, activation='softmax'))

    model.compile(adam_v2.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#
model = myModel()
print(model.summary())
#
#开始训练模型
history = model.fit(dataGen.flow(X_train, y_train,
                              batch_size=batchSizeVal), #每次梯度更新的样本数。未指定，默认为32
                              steps_per_epoch=stepsPerEpochVal, #一个epoch包含的步数
                              epochs=epochsVal, #训练模型迭代次数
                              validation_data=(X_validation, y_validation), #用来评估损失，以及在每轮结束时的任何模型度量指标。
                              shuffle=1) #是否在每轮迭代之前混洗数据 布尔数值

model.save("my_model")

# 绘制训练结果
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()





