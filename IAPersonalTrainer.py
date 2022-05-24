import cv2, time, imutils, numpy as np

from classes import PoseDetector



cap = cv2.VideoCapture("./videos/pose9.mp4")

detector = PoseDetector()

count = dir = currentTime = previousTime = 0



while True :
    success, img = cap.read()

    img = imutils.resize(img, height=720)
    img, lmList = detector.findPosition(img, False)

    if len(lmList) > 0 :
        angle = detector.findAngle(img, 20, 14, 12)
        per = np.interp(angle, (40, 150), (0, 100))
        perBar = np.interp(angle, (40, 150), (100, 650))

        if per == 100 :
            if dir == 0 :
                count += 0.5
                dir = 1

        elif per == 0  :
            if dir == 1 :
                count += 0.5
                dir = 0



        cv2.rectangle(img, (350, 100), (390, 650), (0, 0, 255), cv2.FILLED)
        cv2.rectangle(img, (350, int(perBar)), (390, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{int(count)}", (355, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime


    cv2.putText(img, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Telona massa", img)
    cv2.waitKey(1)