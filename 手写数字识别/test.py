import numpy as np
import cv2
import pickle
from keras.models import load_model
 
########### 参数设置 ##############
#摄像头窗口大小设置
width = 640
height = 480
threshold = 0.8 # 最小置信度
#####################################
 
#### 创建摄像头窗口
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)
 
#### 加载训练好的模型
# pickle_in = open("model_trained_10.p","rb")
# model1 = pickle.load(pickle_in)

model1 = load_model("my_model")


#### 图片预处理
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

#测试图片
def test_img():
    img_original = cv2.imread("test_img.jpg")
    img = cv2.resize(img_original,(32,32))
    img = preProcessing(img)
    img = img.reshape(1, 32, 32, 1)
    predict = model1.predict(img)
    classno = np.argmax(predict, axis=1)

    probVal = np.amax(predict)
    print("预测结果为:",classno, "准确率:",probVal)
    cv2.imshow("Original Image", img_original)
    cv2.waitKey(0)



#测试摄像头
def test_cam():
    while True:
        success, imgOriginal = cap.read()
        img = np.asarray(imgOriginal)
        img = cv2.resize(img,(32,32))
        img = preProcessing(img)
        #cv2.imshow("Processsed Image",img)
        img = img.reshape(1,32,32,1)

        predict = model1.predict(img)
        classno = np.argmax(predict,axis=1)

        probVal= np.amax(predict)
        print(classno,probVal)

        if probVal> threshold:
            cv2.putText(imgOriginal,str(classno) + "   "+str(probVal),
                        (50,50),cv2.FONT_HERSHEY_COMPLEX,
                        2,(0,0,255),1)

        cv2.imshow("Original Image",imgOriginal)
        if cv2.waitKey(1) & 0xFF == 27:
            break

test_img()