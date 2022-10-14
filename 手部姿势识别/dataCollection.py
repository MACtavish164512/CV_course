import os
import time

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

#创建文件夹
# for i in range(1,11):
#     os.makedirs('./data/'+ str(i) )

offset = 20
imgsize = 300
counter = 0

#调用摄像头
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success , img = cap.read()
    hands , img = detector.findHands(img)

    #打印手部信息 裁剪图片
    if hands:
        hands=hands[0]
        x,y,w,h = hands['bbox']
        # print(hands['bbox'])
        Y1,Y2,X1,X2 =y-offset,y+h+offset,x-offset,x+w+offset
        if Y1 <= 0:
            Y1 = 0
        if X1 <= 0:
            X1 = 0
        imgCrop = img[Y1:Y2,X1:X2]
        imgCropshape = imgCrop.shape

        imgWhite = np.ones((imgsize,imgsize,3),np.uint8)*255

        Ratio = h/w
        if Ratio > 1:
            k = imgsize / h
            width = math.ceil(k*w) #无论3.2还是3.5 都会变成4

            imgResize = cv2.resize(imgCrop,(width,imgsize))
            imgResizeShape = imgResize.shape
            center = math.ceil((imgsize - width)/2)
            imgWhite[:, center:width+center] = imgResize


        else:
            k = imgsize / w
            height = math.ceil(k * h)  # 无论3.2还是3.5 都会变成4

            imgResize = cv2.resize(imgCrop, (imgsize,height))
            imgResizeShape = imgResize.shape
            center = math.ceil((imgsize - height) / 2)
            imgWhite[center:center+height, :] = imgResize

        cv2.imshow("imgWhite", imgWhite)
        #cv2.imshow("imgCrop",imgCrop)

    cv2.imshow("input",img)

    key = cv2.waitKey(1);
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'data/10/Image_{time.time()}.jpg',imgWhite)
        print(counter)

    if cv2.waitKey(1) & 0xFF == 27:
        break


