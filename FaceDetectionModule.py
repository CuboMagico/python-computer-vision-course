import cv2, time, imutils, mediapipe as mp


class FaceDetector () :
    
    
    def __init__(self, detectionCon=0.5) :
        self.detectionCon = detectionCon

        self.mpDraw = mp.solutions.drawing_utils
        mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = mpFaceDetection.FaceDetection(self.detectionCon)


    def findFace (self, img, draw=True) :
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxes = []

        if self.results.detections :
            for id, detection in enumerate(self.results.detections) :
                bboxC = detection.location_data.relative_bounding_box

                ih, iw, ic = img.shape


                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id, bbox, detection.score])

                if draw :
                    img = self.fencyDraw(img, bbox)
                    cv2.putText(img, f"{int(detection.score[0] * 100)} %", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

        return img, bboxes


    def fencyDraw (self, img, bbox, l=30, t=5) :
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255,0,255), 1)


        # Top left corner
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        # Top right corner
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        # Bottom left corner
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # Bottom right corner
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img




def main () :
    cap = cv2.VideoCapture("./videos/pose4.mp4")

    currentTime = previousTime = 0

    detector = FaceDetector()

    while True :

        success, img = cap.read()

        img = imutils.resize(img, height=1080)

        img, bboxes = detector.findFace(img)

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__" :
    main()