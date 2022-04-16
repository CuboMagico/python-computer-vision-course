import cv2, time
import mediapipe as mp
import imutils

cap = cv2.VideoCapture("videos/pose1.mp4")


currentTime = previousTime = 0

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

while True :

    success, img = cap.read()
    img = imutils.resize(img, height=720)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    if results.pose_landmarks :
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark) :
            h, w, c = img.shape

            cx, cy = int(lm.x*w), int(lm.y*h)

            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)



    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
