import numpy as np
import cv2
import pickle
from keras.models import load_model

#设置摄像头窗口的大小
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.85  #置信度阈值
font = cv2.FONT_HERSHEY_SIMPLEX

# 启动摄像头
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

#加载模型
model = load_model("model.h5")

#图片的预处理 把图片转换为灰度图 and 图像均衡化
def preProcess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转换为灰度图
    img = cv2.equalizeHist(img) #均衡化
    img = img/255 #归一化
    return img

def getCalssName(classNo):
    if classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vechiles'
    elif classNo == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'


while True:
    #读取摄像头数据
    success , imgOrignal = cap.read()

    img = np.asarray(imgOrignal)
    img = cv2.resize(img,(32,32))
    img = preProcess(img)
    img = img.reshape(1,32,32,1)

    cv2.putText(imgOrignal,"CLASS:",(20,35),font,0.75,(0,255,255),2,cv2.LINE_AA)
    cv2.putText(imgOrignal, "CONFIDENCE:", (20, 75), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    #预测
    prediction = model.predict(img)
    classIndex = np.argmax(prediction,axis=1)
    confidence = np.amax(prediction)

    if confidence > threshold:
        cv2.putText(imgOrignal,str(getCalssName(classIndex)),(120,35),font,0.75,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(imgOrignal,str(round(confidence*100,2)),(180,75),font,0.75,(0,0,255),2,cv2.LINE_AA)

    cv2.imshow("Result",imgOrignal)

    if cv2.waitKey(1) & 0xFF == 27:
        break
