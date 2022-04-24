import cv2, time, imutils, mediapipe as mp


cap = cv2.VideoCapture(0)

previousTime = currentTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)


while True :

    success, img = cap.read()
    img = imutils.resize(img, height=720)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks :
        for faceLms in results.multi_face_landmarks :
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)