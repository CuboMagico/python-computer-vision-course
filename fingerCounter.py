import time, cv2, imutils, mediapipe as mp
from requests import delete

from classes import HandDetector



cap = cv2.VideoCapture(0)

currentTime = previousTime = 0

detector = HandDetector()

counter = 0


while True :
    success, img = cap.read()
    img = imutils.resize(img, height=720)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) > 0 :

        fingers = []

        if id == 20 :
            if lmList[4][1] < lmList[2][1] :
                fingers.append(1)
            
            else :
                fingers.append(0)


        for id in range(8, 21, 4) :
            if lmList[id][2] < lmList[id - 2][2] :
                fingers.append(1)

            else :
                fingers.append(0)

        
        counter = fingers.count(1)


    cv2.putText(img, f"Dedos: {counter}", (50, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, f"FPS: {int(fps)}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


    cv2.imshow("Image", img)
    cv2.waitKey(1)