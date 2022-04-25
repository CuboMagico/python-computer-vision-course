import cv2, time, imutils, numpy as np
from matplotlib.pyplot import sca
from classes import HandDetector

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)

currentTime = previousTime = 0

detector = HandDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

minVol = volume.GetVolumeRange()[0]
maxVol = volume.GetVolumeRange()[1]
vol = volScale = 0
volBar = 150


while True :
    success, img = cap.read()
    img = imutils.resize(img, height=720)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) > 0 :

        x4, y4 = lmList[4][1], lmList[4][2]
        x8, y8 = lmList[8][1], lmList[8][2]

        xm, ym = (x4 + x8) // 2, (y4 + y8) // 2

        cv2.circle(img, (x4, y4), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x8, y8), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (xm, ym), 15, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (x4, y4), (x8, y8), (255, 0, 255), 2)

        lenght = ((x4 - x8) ** 2 + (y4 - y8) ** 2) ** (1 / 2)

        if lenght < 50 :
            cv2.circle(img, (xm, ym), 15, (0, 0, 255), cv2.FILLED)


        vol = np.interp(lenght, [50, 300], [minVol, maxVol])
        volBar = np.interp(lenght, [50, 300], [400, 150])
        volScale = np.interp(lenght, [50, 300], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

    
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, f"FPS: {int(fps)}", (50, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f"{int(volScale)} %", (50, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Telona", img)
    cv2.waitKey(1)