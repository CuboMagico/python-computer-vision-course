import os, numpy as np, mediapipe as mp, time, cv2
from classes import HandDetector


cap = cv2.VideoCapture(0)

detector = HandDetector()
imgCanvas = np.zeros((480, 640, 3), np.uint8) # A resolução do canvas deve ser igual a resolução da sua web cam.

drawColor = ()

while True :
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)


    cv2.rectangle(img, (0, 0), (800, 100), (255, 0, 0), cv2.FILLED)
    
    cv2.putText(img, "Red", (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(img, "Green", (150, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img, "Purple", (300, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.putText(img, "Eraser", (450, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


    if (len(lmList) > 0) :
        
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        
        fingers = detector.fingersUp()

        if fingers[0] and fingers[1] :

            if y1 < 125 :
                if 0 < x1 < 100 :
                    drawColor = (0, 0, 255)

                elif 150 < x1 < 200 :
                    drawColor = (0, 255, 0)

                elif 300 < x1 < 350 :
                    drawColor = (255, 0, 255)
                
                elif 450 < x1 < 500 :
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)


        elif fingers[0] :
            if y1 > 125 :
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            
                cv2.circle(imgCanvas, (x1, y1), 15, drawColor, cv2.FILLED)


    imgGrey = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    

    cv2.imshow("Telona", img)
    cv2.waitKey(1)