import cv2, time, imutils, mediapipe as mp


cap = cv2.VideoCapture("./videos/pose1.mp4")

currentTime = previousTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()

while True :

    success, img = cap.read()

    img = imutils.resize(img, height=720)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)

    if results.detections :
        for id, detection in enumerate(results.detections) :
            bboxC = detection.location_data.relative_bounding_box

            ih, iw, ic = img.shape

            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, str(int(detection.score[0] * 100)), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)


    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
