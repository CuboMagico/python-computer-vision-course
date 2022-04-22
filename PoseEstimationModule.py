import cv2, time
import mediapipe as mp
import imutils


class PoseDetector () :

    def __init__(self, mode=False, upperBody=False, complexity=1, smoothness=True, detectionCon=0.5, trackCon=0.5) :
        self.mode = mode
        self.upperBody = upperBody
        self.complexity = complexity
        self.smoothness = smoothness
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.complexity, self.smoothness, self.detectionCon, self.trackCon)

    

    def findPose (self, img, draw=True) :

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks :
            if draw :
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img
                 

    def findPosition (self, img, draw=True) :

        lmList = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        for id, lm in enumerate(self.results.pose_landmarks.landmark) :
            h, w, c = img.shape

            cx, cy = int(lm.x*w), int(lm.y*h)

            lmList.append([id, cx, cy])

            if draw and id :
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList
   


def main () :
    cap = cv2.VideoCapture("videos/pose4.mp4")

    currentTime = previousTime = 0

    detector = PoseDetector()
    
    while True :

        success, img = cap.read()
        img = imutils.resize(img, height=720)

        
        lmList = detector.findPosition(img)

        if len(lmList) != 0 :
            print(lmList)

        currentTime = time.time()
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__" :
    main()